import {
  EmbeddingModelV2,
  LanguageModelV2,
  ProviderV2,
  ImageModelV2,
} from '@ai-sdk/provider';
import {
  FetchFunction,
  generateId,
  loadApiKey,
  withoutTrailingSlash,
  withUserAgentSuffix,
} from '@ai-sdk/provider-utils';
import { VERSION } from './version';
import { GoogleGenerativeAIEmbeddingModel } from './google-generative-ai-embedding-model';
import { GoogleGenerativeAIEmbeddingModelId } from './google-generative-ai-embedding-options';
import { GoogleGenerativeAILanguageModel } from './google-generative-ai-language-model';
import { GoogleGenerativeAIModelId } from './google-generative-ai-options';
import { googleTools } from './google-tools';

import {
  GoogleGenerativeAIImageSettings,
  GoogleGenerativeAIImageModelId,
} from './google-generative-ai-image-settings';
import { GoogleGenerativeAIImageModel } from './google-generative-ai-image-model';

export interface GoogleGenerativeAIProvider extends ProviderV2 {
  (modelId: GoogleGenerativeAIModelId): LanguageModelV2;

  languageModel(modelId: GoogleGenerativeAIModelId): LanguageModelV2;

  chat(modelId: GoogleGenerativeAIModelId): LanguageModelV2;

  /**
Creates a model for image generation.
 */
  image(
    modelId: GoogleGenerativeAIImageModelId,
    settings?: GoogleGenerativeAIImageSettings,
  ): ImageModelV2;

  /**
   * @deprecated Use `chat()` instead.
   */
  generativeAI(modelId: GoogleGenerativeAIModelId): LanguageModelV2;

  /**
@deprecated Use `textEmbedding()` instead.
   */
  embedding(
    modelId: GoogleGenerativeAIEmbeddingModelId,
  ): EmbeddingModelV2<string>;

  textEmbedding(
    modelId: GoogleGenerativeAIEmbeddingModelId,
  ): EmbeddingModelV2<string>;

  textEmbeddingModel(
    modelId: GoogleGenerativeAIEmbeddingModelId,
  ): EmbeddingModelV2<string>;

  tools: typeof googleTools;
}

export interface GoogleGenerativeAIProviderSettings {
  /**
Use a different URL prefix for API calls, e.g. to use proxy servers.
The default prefix is `https://cloudcode-pa.googleapis.com`.
   */
  baseURL?: string;

  /**
Google Cloud access token used for Bearer authentication.
It defaults to the `GOOGLE_CLOUD_ACCESS_TOKEN` environment variable.
   */
  apiKey?: string;

  /**
Custom headers to include in the requests.
     */
  headers?: Record<string, string | undefined>;

  /**
Custom fetch implementation. You can use it as a middleware to intercept requests,
or to provide a custom fetch implementation for e.g. testing.
    */
  fetch?: FetchFunction;

  /**
Optional function to generate a unique ID for each request.
     */
  generateId?: () => string;

  /**
   * Custom provider name
   * Defaults to 'google.generative-ai'.
   */
  name?: string;
}

/**
Create a Google Generative AI provider instance.
 */
export function createGoogleGenerativeAI(
  options: GoogleGenerativeAIProviderSettings = {},
): GoogleGenerativeAIProvider {
  const baseURL =
    withoutTrailingSlash(options.baseURL) ??
    'https://cloudcode-pa.googleapis.com';

  const providerName = options.name ?? 'cloudassist';

  const getHeaders = () =>
    withUserAgentSuffix(
      {
        Authorization: `Bearer ${loadApiKey({
          apiKey: options.apiKey,
          environmentVariableName: 'GOOGLE_CLOUD_ACCESS_TOKEN',
          description: 'Google Cloud Access Token',
        })}`,
        ...options.headers,
      },
      `ai-sdk/google-cloudassist/${VERSION}`,
    );

  const createChatModel = (modelId: GoogleGenerativeAIModelId) =>
    new GoogleGenerativeAILanguageModel(modelId, {
      provider: providerName,
      baseURL,
      headers: getHeaders,
      generateId: options.generateId ?? generateId,
      supportedUrls: () => ({
        '*': [
          // Google Generative Language "files" endpoint
          // e.g. https://generativelanguage.googleapis.com/v1beta/files/...
          new RegExp(`^${baseURL}/files/.*$`),
          // YouTube URLs (public or unlisted videos)
          new RegExp(
            `^https://(?:www\\.)?youtube\\.com/watch\\?v=[\\w-]+(?:&[\\w=&.-]*)?$`,
          ),
          new RegExp(`^https://youtu\\.be/[\\w-]+(?:\\?[\\w=&.-]*)?$`),
        ],
      }),
      fetch: options.fetch,
    });

  const createEmbeddingModel = (modelId: GoogleGenerativeAIEmbeddingModelId) =>
    new GoogleGenerativeAIEmbeddingModel(modelId, {
      provider: providerName,
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const createImageModel = (
    modelId: GoogleGenerativeAIImageModelId,
    settings: GoogleGenerativeAIImageSettings = {},
  ) =>
    new GoogleGenerativeAIImageModel(modelId, settings, {
      provider: providerName,
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const provider = function (modelId: GoogleGenerativeAIModelId) {
    if (new.target) {
      throw new Error(
        'The Google Generative AI model function cannot be called with the new keyword.',
      );
    }

    return createChatModel(modelId);
  };

  provider.languageModel = createChatModel;
  provider.chat = createChatModel;
  provider.generativeAI = createChatModel;
  provider.embedding = createEmbeddingModel;
  provider.textEmbedding = createEmbeddingModel;
  provider.textEmbeddingModel = createEmbeddingModel;
  provider.image = createImageModel;
  provider.imageModel = createImageModel;
  provider.tools = googleTools;
  return provider as GoogleGenerativeAIProvider;
}

/**
Default Google Generative AI provider instance.
 */
export const google = createGoogleGenerativeAI();
