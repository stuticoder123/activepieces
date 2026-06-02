```ts
import {
  createAction,
  Property,
  StoreScope,
} from '@activepieces/pieces-framework';

import { propsValidation } from '@activepieces/pieces-common';

import OpenAI from 'openai';
import { z } from 'zod';

import { openaiAuth } from '../auth';

import {
  calculateMessagesTokenSize,
  exceedsHistoryLimit,
  notLLMs,
  reduceContextSize,
} from '../common/common';

// ========================================
// Constants
// ========================================

const DEFAULT_MODEL = 'gpt-4.1-mini';

const ALLOWED_ROLES = ['system', 'user', 'assistant'] as const;

const MAX_MEMORY_KEY_LENGTH = 128;

const MAX_PROMPT_LENGTH = 100000;

// ========================================
// Types
// ========================================

type ChatRole = (typeof ALLOWED_ROLES)[number];

type ChatMessage = {
  role: ChatRole;
  content: string;
};

// ========================================
// Helpers
// ========================================

const validateRoles = (roles: unknown): ChatMessage[] => {
  if (!Array.isArray(roles)) {
    return [];
  }

  return roles.map((item: any) => {
    if (!ALLOWED_ROLES.includes(item.role)) {
      throw new Error(
        `Invalid role "${item.role}". Allowed roles: ${ALLOWED_ROLES.join(
          ', '
        )}`
      );
    }

    if (!item.content || typeof item.content !== 'string') {
      throw new Error('Each role must contain valid text content.');
    }

    return {
      role: item.role,
      content: item.content,
    };
  });
};

const normalizeOpenAIError = (error: unknown): never => {
  if (error instanceof OpenAI.APIError) {
    switch (error.status) {
      case 401:
        throw new Error('Invalid OpenAI API key.');

      case 429:
        throw new Error(
          'OpenAI rate limit exceeded. Please retry later.'
        );

      case 500:
        throw new Error(
          'OpenAI internal server error. Please try again.'
        );

      default:
        throw new Error(
          `OpenAI API Error: ${error.message}`
        );
    }
  }

  if (error instanceof Error) {
    throw new Error(error.message);
  }

  throw new Error('Unknown OpenAI error occurred.');
};

// ========================================
// Action
// ========================================

export const askOpenAI = createAction({
  auth: openaiAuth,

  name: 'ask_chatgpt',

  displayName: 'Ask ChatGPT',

  description:
    'Ask ChatGPT questions with optional memory and role-based instructions.',

  props: {
    model: Property.Dropdown({
      auth: openaiAuth,

      displayName: 'Model',

      required: true,

      description:
        'Select the OpenAI model used to generate the response.',

      refreshers: [],

      defaultValue: DEFAULT_MODEL,

      options: async ({ auth }) => {
        if (!auth) {
          return {
            disabled: true,
            placeholder: 'Enter your OpenAI API key first.',
            options: [],
          };
        }

        try {
          const openai = new OpenAI({
            apiKey: auth.secret_text,
          });

          const response = await openai.models.list();

          const models = response.data.filter(
            (model) => !notLLMs.includes(model.id)
          );

          return {
            disabled: false,
            options: models.map((model) => ({
              label: model.id,
              value: model.id,
            })),
          };
        } catch {
          return {
            disabled: true,
            placeholder: 'Unable to load models. Check your API key.',
            options: [],
          };
        }
      },
    }),

    prompt: Property.LongText({
      displayName: 'Question',

      description: 'The prompt sent to ChatGPT.',

      required: true,
    }),

    temperature: Property.Number({
      displayName: 'Temperature',

      description:
        'Controls response randomness. Lower values are more deterministic.',

      required: false,

      defaultValue: 1,
    }),

    maxTokens: Property.Number({
      displayName: 'Maximum Tokens',

      description:
        'Maximum number of tokens generated in the response.',

      required: true,

      defaultValue: 2048,
    }),

    topP: Property.Number({
      displayName: 'Top P',

      description:
        'Controls nucleus sampling probability threshold.',

      required: false,

      defaultValue: 1,
    }),

    frequencyPenalty: Property.Number({
      displayName: 'Frequency Penalty',

      description:
        'Penalizes repeated token usage.',

      required: false,

      defaultValue: 0,
    }),

    presencePenalty: Property.Number({
      displayName: 'Presence Penalty',

      description:
        'Encourages discussing new topics.',

      required: false,

      defaultValue: 0,
    }),

    memoryKey: Property.ShortText({
      displayName: 'Memory Key',

      description:
        'Optional shared memory key for preserving chat history.',

      required: false,
    }),

    roles: Property.Json({
      displayName: 'Roles',

      description:
        'Optional role instructions for system/user/assistant context.',

      required: false,

      defaultValue: [
        {
          role: 'system',
          content: 'You are a helpful assistant.',
        },
      ],
    }),
  },

  async run({ auth, propsValue, store }) {
    try {
      await propsValidation.validateZod(propsValue, {
        temperature: z.number().min(0).max(2).optional(),

        topP: z.number().min(0).max(1).optional(),

        frequencyPenalty: z.number().min(-2).max(2).optional(),

        presencePenalty: z.number().min(-2).max(2).optional(),

        memoryKey: z
          .string()
          .max(MAX_MEMORY_KEY_LENGTH)
          .optional(),
      });

      const {
        model,
        prompt,
        temperature,
        maxTokens,
        topP,
        frequencyPenalty,
        presencePenalty,
        memoryKey,
      } = propsValue;

      if (!prompt?.trim()) {
        throw new Error('Prompt cannot be empty.');
      }

      if (prompt.length > MAX_PROMPT_LENGTH) {
        throw new Error(
          'Prompt exceeds maximum supported length.'
        );
      }

      const openai = new OpenAI({
        apiKey: auth.secret_text,
      });

      let messageHistory: ChatMessage[] =
        memoryKey
          ? (await store.get(
              memoryKey,
              StoreScope.PROJECT
            )) ?? []
          : [];

      messageHistory.push({
        role: 'user',
        content: prompt,
      });

      const roles = validateRoles(propsValue.roles);

      const completion =
        await openai.chat.completions.create({
          model,

          messages: [...roles, ...messageHistory],

          temperature,

          top_p: topP,

          frequency_penalty: frequencyPenalty,

          presence_penalty: presencePenalty,

          max_completion_tokens: maxTokens,
        });

      const assistantMessage =
        completion.choices?.[0]?.message;

      if (!assistantMessage?.content) {
        throw new Error(
          'OpenAI returned an empty response.'
        );
      }

      messageHistory.push({
        role: 'assistant',
        content: assistantMessage.content,
      });

      const tokenLength =
        await calculateMessagesTokenSize(
          messageHistory,
          model
        );

      if (
        memoryKey &&
        exceedsHistoryLimit(
          tokenLength,
          model,
          maxTokens
        )
      ) {
        messageHistory = await reduceContextSize(
          messageHistory,
          model,
          maxTokens
        );
      }

      if (memoryKey) {
        await store.put(
          memoryKey,
          messageHistory,
          StoreScope.PROJECT
        );
      }

      return assistantMessage.content;
    } catch (error) {
      normalizeOpenAIError(error);
    }
  },
});
```
