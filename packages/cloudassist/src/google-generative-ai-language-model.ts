import {
  LanguageModelV2,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2Source,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
  SharedV2ProviderMetadata,
} from '@ai-sdk/provider';
import {
  FetchFunction,
  InferValidator,
  ParseResult,
  Resolvable,
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  generateId,
  lazySchema,
  parseProviderOptions,
  postJsonToApi,
  resolve,
  zodSchema,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import { convertJSONSchemaToOpenAPISchema } from './convert-json-schema-to-openapi-schema';
import { convertToGoogleGenerativeAIMessages } from './convert-to-google-generative-ai-messages';
import { getModelPath } from './get-model-path';
import { googleFailedResponseHandler } from './google-error';
import { GoogleGenerativeAIContentPart } from './google-generative-ai-prompt';
import {
  GoogleGenerativeAIModelId,
  googleGenerativeAIProviderOptions,
} from './google-generative-ai-options';
import { prepareTools, isClaudeModel } from './google-prepare-tools';
import { mapGoogleGenerativeAIFinishReason } from './map-google-generative-ai-finish-reason';

type GoogleGenerativeAIConfig = {
  provider: string;
  baseURL: string;
  headers: Resolvable<Record<string, string | undefined>>;
  fetch?: FetchFunction;
  generateId: () => string;

  /**
   * The supported URLs for the model.
   */
  supportedUrls?: () => LanguageModelV2['supportedUrls'];
};

export class GoogleGenerativeAILanguageModel implements LanguageModelV2 {
  readonly specificationVersion = 'v2';

  readonly modelId: GoogleGenerativeAIModelId;

  private readonly config: GoogleGenerativeAIConfig;
  private readonly generateId: () => string;

  constructor(
    modelId: GoogleGenerativeAIModelId,
    config: GoogleGenerativeAIConfig,
  ) {
    this.modelId = modelId;
    this.config = config;
    this.generateId = config.generateId ?? generateId;
  }

  get provider(): string {
    return this.config.provider;
  }

  get supportedUrls() {
    return this.config.supportedUrls?.() ?? {};
  }

  private async getArgs({
    prompt,
    maxOutputTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    stopSequences,
    responseFormat,
    seed,
    tools,
    toolChoice,
    providerOptions,
  }: Parameters<LanguageModelV2['doGenerate']>[0]) {
    const warnings: LanguageModelV2CallWarning[] = [];

    const googleOptions = await parseProviderOptions({
      provider: 'google',
      providerOptions,
      schema: googleGenerativeAIProviderOptions,
    });

    // Add warning if Vertex rag tools are used with a non-Vertex Google provider
    if (
      tools?.some(
        tool =>
          tool.type === 'provider-defined' &&
          tool.id === 'google.vertex_rag_store',
      ) &&
      !this.config.provider.startsWith('google.vertex.')
    ) {
      warnings.push({
        type: 'other',
        message:
          "The 'vertex_rag_store' tool is only supported with the Google Vertex provider " +
          'and might not be supported or could behave unexpectedly with the current Google provider ' +
          `(${this.config.provider}).`,
      });
    }

    const isGemmaModel = this.modelId.toLowerCase().startsWith('gemma-');

    const { contents, systemInstruction } = convertToGoogleGenerativeAIMessages(
      prompt,
      { isGemmaModel, modelId: this.modelId },
    );

    const {
      tools: googleTools,
      toolConfig: googleToolConfig,
      toolWarnings,
    } = prepareTools({
      tools,
      toolChoice,
      modelId: this.modelId,
    });

    return {
      args: {
        generationConfig: {
          // standardized settings:
          maxOutputTokens,
          temperature,
          topK,
          topP,
          frequencyPenalty,
          presencePenalty,
          stopSequences,
          seed,

          // response format:
          responseMimeType:
            responseFormat?.type === 'json' ? 'application/json' : undefined,
          responseSchema:
            responseFormat?.type === 'json' &&
            responseFormat.schema != null &&
            // Google GenAI does not support all OpenAPI Schema features,
            // so this is needed as an escape hatch:
            // TODO convert into provider option
            (googleOptions?.structuredOutputs ?? true)
              ? convertJSONSchemaToOpenAPISchema(responseFormat.schema)
              : undefined,
          ...(googleOptions?.audioTimestamp && {
            audioTimestamp: googleOptions.audioTimestamp,
          }),

          // provider options:
          responseModalities: googleOptions?.responseModalities,
          thinkingConfig: googleOptions?.thinkingConfig,
          ...(googleOptions?.imageConfig && {
            imageConfig: googleOptions.imageConfig,
          }),
          ...(googleOptions?.mediaResolution && {
            mediaResolution: googleOptions.mediaResolution,
          }),
        },
        contents,
        systemInstruction: isGemmaModel ? undefined : systemInstruction,
        safetySettings: googleOptions?.safetySettings,
        tools: googleTools,
        toolConfig: googleOptions?.retrievalConfig
          ? {
              ...googleToolConfig,
              retrievalConfig: googleOptions.retrievalConfig,
            }
          : googleToolConfig,
        cachedContent: googleOptions?.cachedContent,
        labels: googleOptions?.labels,
        // Cloud Assist: sessionId goes inside the request
        ...(googleOptions?.sessionId && { sessionId: googleOptions.sessionId }),
      },
      warnings: [...warnings, ...toolWarnings],
      // Cloud Assist specific options (used to wrap the request)
      cloudAssistOptions: {
        projectId: googleOptions?.projectId,
        requestType: googleOptions?.requestType,
        userAgent: googleOptions?.userAgent,
        requestId: googleOptions?.requestId,
      },
    };
  }

  async doGenerate(
    options: Parameters<LanguageModelV2['doGenerate']>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2['doGenerate']>>> {
    const { args, warnings, cloudAssistOptions } = await this.getArgs(options);

    // Wrap request for Cloud Assist API
    const wrappedRequest = {
      project: cloudAssistOptions.projectId,
      model: this.modelId,
      request: args,
      ...(cloudAssistOptions.requestType && {
        requestType: cloudAssistOptions.requestType,
      }),
      userAgent: cloudAssistOptions.userAgent ?? 'ai-sdk',
      requestId:
        cloudAssistOptions.requestId ??
        `ai-sdk-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
    };

    const body = JSON.stringify(wrappedRequest);

    const mergedHeaders = combineHeaders(
      await resolve(this.config.headers),
      options.headers,
    );

    // Use SSE endpoint and collect all chunks
    const { responseHeaders, value: response } = await postJsonToApi({
      url: `${this.config.baseURL}/v1internal:streamGenerateContent?alt=sse`,
      headers: mergedHeaders,
      body: wrappedRequest,
      failedResponseHandler: googleFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(chunkSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    // Collect all chunks from the SSE stream
    const content: Array<LanguageModelV2Content> = [];
    let finishReason: LanguageModelV2FinishReason = 'unknown';
    let usageMetadata: ChunkSchema['usageMetadata'] | undefined;
    let promptFeedback: ChunkSchema['promptFeedback'] | undefined;
    let groundingMetadata:
      | NonNullable<
          NonNullable<ChunkSchema['candidates']>[number]['groundingMetadata']
        >
      | undefined;
    let urlContextMetadata:
      | NonNullable<
          NonNullable<ChunkSchema['candidates']>[number]['urlContextMetadata']
        >
      | undefined;
    let safetyRatings:
      | NonNullable<
          NonNullable<ChunkSchema['candidates']>[number]['safetyRatings']
        >
      | undefined;

    // Track text/reasoning blocks for aggregation
    let currentTextContent = '';
    let currentTextThoughtSignature: string | undefined;
    let currentReasoningContent = '';
    let currentReasoningThoughtSignature: string | undefined;
    let isInReasoning = false;
    let lastCodeExecutionToolCallId: string | undefined;

    const useToolCallId = isClaudeModel(this.modelId);

    const reader = response.getReader();
    try {
      while (true) {
        const { done, value: chunk } = await reader.read();
        if (done) break;

        if (!chunk.success) continue;

        const value = chunk.value;

        // Unwrap the response from Cloud Assist format
        const responseData = value.response ?? value;

        if (responseData.usageMetadata) {
          usageMetadata = responseData.usageMetadata;
        }
        if (responseData.promptFeedback) {
          promptFeedback = responseData.promptFeedback;
        }

        const candidate = responseData.candidates?.[0];
        if (!candidate) continue;

        if (candidate.groundingMetadata) {
          groundingMetadata = candidate.groundingMetadata;
        }
        if (candidate.urlContextMetadata) {
          urlContextMetadata = candidate.urlContextMetadata;
        }
        if (candidate.safetyRatings) {
          safetyRatings = candidate.safetyRatings;
        }

        const parts = candidate.content?.parts ?? [];
        for (const part of parts) {
          if ('executableCode' in part && part.executableCode?.code) {
            // Flush any pending text/reasoning
            if (currentTextContent) {
              content.push({
                type: 'text',
                text: currentTextContent,
                providerMetadata: currentTextThoughtSignature
                  ? { google: { thoughtSignature: currentTextThoughtSignature } }
                  : undefined,
              });
              currentTextContent = '';
              currentTextThoughtSignature = undefined;
            }
            if (currentReasoningContent) {
              content.push({
                type: 'reasoning',
                text: currentReasoningContent,
                providerMetadata: currentReasoningThoughtSignature
                  ? {
                      google: {
                        thoughtSignature: currentReasoningThoughtSignature,
                      },
                    }
                  : undefined,
              });
              currentReasoningContent = '';
              currentReasoningThoughtSignature = undefined;
            }

            const toolCallId = this.config.generateId();
            lastCodeExecutionToolCallId = toolCallId;
            content.push({
              type: 'tool-call',
              toolCallId,
              toolName: 'code_execution',
              input: JSON.stringify(part.executableCode),
              providerExecuted: true,
            });
          } else if (
            'codeExecutionResult' in part &&
            part.codeExecutionResult
          ) {
            content.push({
              type: 'tool-result',
              toolCallId: lastCodeExecutionToolCallId!,
              toolName: 'code_execution',
              result: {
                outcome: part.codeExecutionResult.outcome,
                output: part.codeExecutionResult.output,
              },
              providerExecuted: true,
            });
            lastCodeExecutionToolCallId = undefined;
          } else if ('text' in part && part.text != null) {
            const isThinking = part.thought === true;

            if (isThinking) {
              // Flush text if switching to reasoning
              if (!isInReasoning && currentTextContent) {
                content.push({
                  type: 'text',
                  text: currentTextContent,
                  providerMetadata: currentTextThoughtSignature
                    ? {
                        google: {
                          thoughtSignature: currentTextThoughtSignature,
                        },
                      }
                    : undefined,
                });
                currentTextContent = '';
                currentTextThoughtSignature = undefined;
              }
              isInReasoning = true;
              currentReasoningContent += part.text;
              if (part.thoughtSignature) {
                currentReasoningThoughtSignature = part.thoughtSignature;
              }
            } else {
              // Flush reasoning if switching to text
              if (isInReasoning && currentReasoningContent) {
                content.push({
                  type: 'reasoning',
                  text: currentReasoningContent,
                  providerMetadata: currentReasoningThoughtSignature
                    ? {
                        google: {
                          thoughtSignature: currentReasoningThoughtSignature,
                        },
                      }
                    : undefined,
                });
                currentReasoningContent = '';
                currentReasoningThoughtSignature = undefined;
              }
              isInReasoning = false;
              currentTextContent += part.text;
              if (part.thoughtSignature) {
                currentTextThoughtSignature = part.thoughtSignature;
              }
            }
          } else if ('functionCall' in part) {
            // Flush any pending text/reasoning
            if (currentTextContent) {
              content.push({
                type: 'text',
                text: currentTextContent,
                providerMetadata: currentTextThoughtSignature
                  ? { google: { thoughtSignature: currentTextThoughtSignature } }
                  : undefined,
              });
              currentTextContent = '';
              currentTextThoughtSignature = undefined;
            }
            if (currentReasoningContent) {
              content.push({
                type: 'reasoning',
                text: currentReasoningContent,
                providerMetadata: currentReasoningThoughtSignature
                  ? {
                      google: {
                        thoughtSignature: currentReasoningThoughtSignature,
                      },
                    }
                  : undefined,
              });
              currentReasoningContent = '';
              currentReasoningThoughtSignature = undefined;
            }

            // Use provided id for Claude models, generate for others
            const toolCallId =
              useToolCallId && part.functionCall.id
                ? part.functionCall.id
                : this.config.generateId();

            content.push({
              type: 'tool-call' as const,
              toolCallId,
              toolName: part.functionCall.name,
              input: JSON.stringify(part.functionCall.args),
              providerMetadata: part.thoughtSignature
                ? { google: { thoughtSignature: part.thoughtSignature } }
                : undefined,
            });
          } else if ('inlineData' in part) {
            // Flush any pending text/reasoning
            if (currentTextContent) {
              content.push({
                type: 'text',
                text: currentTextContent,
                providerMetadata: currentTextThoughtSignature
                  ? { google: { thoughtSignature: currentTextThoughtSignature } }
                  : undefined,
              });
              currentTextContent = '';
              currentTextThoughtSignature = undefined;
            }
            if (currentReasoningContent) {
              content.push({
                type: 'reasoning',
                text: currentReasoningContent,
                providerMetadata: currentReasoningThoughtSignature
                  ? {
                      google: {
                        thoughtSignature: currentReasoningThoughtSignature,
                      },
                    }
                  : undefined,
              });
              currentReasoningContent = '';
              currentReasoningThoughtSignature = undefined;
            }

            content.push({
              type: 'file' as const,
              data: part.inlineData.data,
              mediaType: part.inlineData.mimeType,
            });
          }
        }

        if (candidate.finishReason) {
          finishReason = mapGoogleGenerativeAIFinishReason({
            finishReason: candidate.finishReason,
            hasToolCalls: content.some(part => part.type === 'tool-call'),
          });
        }
      }
    } finally {
      reader.releaseLock();
    }

    // Flush any remaining text/reasoning
    if (currentTextContent) {
      content.push({
        type: 'text',
        text: currentTextContent,
        providerMetadata: currentTextThoughtSignature
          ? { google: { thoughtSignature: currentTextThoughtSignature } }
          : undefined,
      });
    }
    if (currentReasoningContent) {
      content.push({
        type: 'reasoning',
        text: currentReasoningContent,
        providerMetadata: currentReasoningThoughtSignature
          ? { google: { thoughtSignature: currentReasoningThoughtSignature } }
          : undefined,
      });
    }

    // Extract sources from grounding metadata
    const sources =
      extractSources({
        groundingMetadata,
        generateId: this.config.generateId,
      }) ?? [];
    for (const source of sources) {
      content.push(source);
    }

    return {
      content,
      finishReason,
      usage: {
        inputTokens: usageMetadata?.promptTokenCount ?? undefined,
        outputTokens: usageMetadata?.candidatesTokenCount ?? undefined,
        totalTokens: usageMetadata?.totalTokenCount ?? undefined,
        reasoningTokens: usageMetadata?.thoughtsTokenCount ?? undefined,
        cachedInputTokens: usageMetadata?.cachedContentTokenCount ?? undefined,
      },
      warnings,
      providerMetadata: {
        google: {
          promptFeedback: promptFeedback ?? null,
          groundingMetadata: groundingMetadata ?? null,
          urlContextMetadata: urlContextMetadata ?? null,
          safetyRatings: safetyRatings ?? null,
          usageMetadata: usageMetadata ?? null,
        },
      },
      request: { body },
      response: {
        headers: responseHeaders,
      },
    };
  }

  async doStream(
    options: Parameters<LanguageModelV2['doStream']>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2['doStream']>>> {
    const { args, warnings, cloudAssistOptions } = await this.getArgs(options);

    // Wrap request for Cloud Assist API
    const wrappedRequest = {
      project: cloudAssistOptions.projectId,
      model: this.modelId,
      request: args,
      ...(cloudAssistOptions.requestType && {
        requestType: cloudAssistOptions.requestType,
      }),
      userAgent: cloudAssistOptions.userAgent ?? 'ai-sdk',
      requestId:
        cloudAssistOptions.requestId ??
        `ai-sdk-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
    };

    const body = JSON.stringify(wrappedRequest);
    const headers = combineHeaders(
      await resolve(this.config.headers),
      options.headers,
    );

    const { responseHeaders, value: response } = await postJsonToApi({
      url: `${this.config.baseURL}/v1internal:streamGenerateContent?alt=sse`,
      headers,
      body: wrappedRequest,
      failedResponseHandler: googleFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(chunkSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    let finishReason: LanguageModelV2FinishReason = 'unknown';
    const usage: LanguageModelV2Usage = {
      inputTokens: undefined,
      outputTokens: undefined,
      totalTokens: undefined,
    };
    let providerMetadata: SharedV2ProviderMetadata | undefined = undefined;

    const generateId = this.config.generateId;
    const useToolCallId = isClaudeModel(this.modelId);
    let hasToolCalls = false;

    // Track active blocks to group consecutive parts of same type
    let currentTextBlockId: string | null = null;
    let currentReasoningBlockId: string | null = null;
    let blockCounter = 0;

    // Track emitted sources to prevent duplicates
    const emittedSourceUrls = new Set<string>();
    // Associates a code execution result with its preceding call.
    let lastCodeExecutionToolCallId: string | undefined;

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<ChunkSchema>,
          LanguageModelV2StreamPart
        >({
          start(controller) {
            controller.enqueue({ type: 'stream-start', warnings });
          },

          transform(chunk, controller) {
            if (options.includeRawChunks) {
              controller.enqueue({ type: 'raw', rawValue: chunk.rawValue });
            }

            if (!chunk.success) {
              controller.enqueue({ type: 'error', error: chunk.error });
              return;
            }

            const value = chunk.value;

            // Unwrap the response from Cloud Assist format
            const responseData = value.response ?? value;

            const usageMetadata = responseData.usageMetadata;

            if (usageMetadata != null) {
              usage.inputTokens = usageMetadata.promptTokenCount ?? undefined;
              usage.outputTokens =
                usageMetadata.candidatesTokenCount ?? undefined;
              usage.totalTokens = usageMetadata.totalTokenCount ?? undefined;
              usage.reasoningTokens =
                usageMetadata.thoughtsTokenCount ?? undefined;
              usage.cachedInputTokens =
                usageMetadata.cachedContentTokenCount ?? undefined;
            }

            const candidate = responseData.candidates?.[0];

            // sometimes the API returns an empty candidates array
            if (candidate == null) {
              return;
            }

            const content = candidate.content;

            const sources = extractSources({
              groundingMetadata: candidate.groundingMetadata,
              generateId,
            });
            if (sources != null) {
              for (const source of sources) {
                if (
                  source.sourceType === 'url' &&
                  !emittedSourceUrls.has(source.url)
                ) {
                  emittedSourceUrls.add(source.url);
                  controller.enqueue(source);
                }
              }
            }

            // Process tool call's parts before determining finishReason to ensure hasToolCalls is properly set
            if (content != null) {
              // Process all parts in a single loop to preserve original order
              const parts = content.parts ?? [];
              for (const part of parts) {
                if ('executableCode' in part && part.executableCode?.code) {
                  const toolCallId = generateId();
                  lastCodeExecutionToolCallId = toolCallId;

                  controller.enqueue({
                    type: 'tool-call',
                    toolCallId,
                    toolName: 'code_execution',
                    input: JSON.stringify(part.executableCode),
                    providerExecuted: true,
                  });

                  hasToolCalls = true;
                } else if (
                  'codeExecutionResult' in part &&
                  part.codeExecutionResult
                ) {
                  // Assumes a result directly follows its corresponding call part.
                  const toolCallId = lastCodeExecutionToolCallId;

                  if (toolCallId) {
                    controller.enqueue({
                      type: 'tool-result',
                      toolCallId,
                      toolName: 'code_execution',
                      result: {
                        outcome: part.codeExecutionResult.outcome,
                        output: part.codeExecutionResult.output,
                      },
                      providerExecuted: true,
                    });
                    // Clear the ID after use.
                    lastCodeExecutionToolCallId = undefined;
                  }
                } else if (
                  'text' in part &&
                  part.text != null &&
                  part.text.length > 0
                ) {
                  if (part.thought === true) {
                    // End any active text block before starting reasoning
                    if (currentTextBlockId !== null) {
                      controller.enqueue({
                        type: 'text-end',
                        id: currentTextBlockId,
                      });
                      currentTextBlockId = null;
                    }

                    // Start new reasoning block if not already active
                    if (currentReasoningBlockId === null) {
                      currentReasoningBlockId = String(blockCounter++);
                      controller.enqueue({
                        type: 'reasoning-start',
                        id: currentReasoningBlockId,
                        providerMetadata: part.thoughtSignature
                          ? {
                              google: {
                                thoughtSignature: part.thoughtSignature,
                              },
                            }
                          : undefined,
                      });
                    }

                    controller.enqueue({
                      type: 'reasoning-delta',
                      id: currentReasoningBlockId,
                      delta: part.text,
                      providerMetadata: part.thoughtSignature
                        ? {
                            google: { thoughtSignature: part.thoughtSignature },
                          }
                        : undefined,
                    });
                  } else {
                    // End any active reasoning block before starting text
                    if (currentReasoningBlockId !== null) {
                      controller.enqueue({
                        type: 'reasoning-end',
                        id: currentReasoningBlockId,
                      });
                      currentReasoningBlockId = null;
                    }

                    // Start new text block if not already active
                    if (currentTextBlockId === null) {
                      currentTextBlockId = String(blockCounter++);
                      controller.enqueue({
                        type: 'text-start',
                        id: currentTextBlockId,
                        providerMetadata: part.thoughtSignature
                          ? {
                              google: {
                                thoughtSignature: part.thoughtSignature,
                              },
                            }
                          : undefined,
                      });
                    }

                    controller.enqueue({
                      type: 'text-delta',
                      id: currentTextBlockId,
                      delta: part.text,
                      providerMetadata: part.thoughtSignature
                        ? {
                            google: { thoughtSignature: part.thoughtSignature },
                          }
                        : undefined,
                    });
                  }
                } else if ('inlineData' in part) {
                  // Process file parts inline to preserve order with text
                  controller.enqueue({
                    type: 'file',
                    mediaType: part.inlineData.mimeType,
                    data: part.inlineData.data,
                  });
                }
              }

              const toolCallDeltas = getToolCallsFromParts({
                parts: content.parts,
                generateId,
                useToolCallId,
              });

              if (toolCallDeltas != null) {
                for (const toolCall of toolCallDeltas) {
                  controller.enqueue({
                    type: 'tool-input-start',
                    id: toolCall.toolCallId,
                    toolName: toolCall.toolName,
                    providerMetadata: toolCall.providerMetadata,
                  });

                  controller.enqueue({
                    type: 'tool-input-delta',
                    id: toolCall.toolCallId,
                    delta: toolCall.args,
                    providerMetadata: toolCall.providerMetadata,
                  });

                  controller.enqueue({
                    type: 'tool-input-end',
                    id: toolCall.toolCallId,
                    providerMetadata: toolCall.providerMetadata,
                  });

                  controller.enqueue({
                    type: 'tool-call',
                    toolCallId: toolCall.toolCallId,
                    toolName: toolCall.toolName,
                    input: toolCall.args,
                    providerMetadata: toolCall.providerMetadata,
                  });

                  hasToolCalls = true;
                }
              }
            }

            if (candidate.finishReason != null) {
              finishReason = mapGoogleGenerativeAIFinishReason({
                finishReason: candidate.finishReason,
                hasToolCalls,
              });

              providerMetadata = {
                google: {
                  promptFeedback: responseData.promptFeedback ?? null,
                  groundingMetadata: candidate.groundingMetadata ?? null,
                  urlContextMetadata: candidate.urlContextMetadata ?? null,
                  safetyRatings: candidate.safetyRatings ?? null,
                },
              };
              if (usageMetadata != null) {
                providerMetadata.google.usageMetadata = usageMetadata;
              }
            }
          },

          flush(controller) {
            // Close any open blocks before finishing
            if (currentTextBlockId !== null) {
              controller.enqueue({
                type: 'text-end',
                id: currentTextBlockId,
              });
            }
            if (currentReasoningBlockId !== null) {
              controller.enqueue({
                type: 'reasoning-end',
                id: currentReasoningBlockId,
              });
            }

            controller.enqueue({
              type: 'finish',
              finishReason,
              usage,
              providerMetadata,
            });
          },
        }),
      ),
      response: { headers: responseHeaders },
      request: { body },
    };
  }
}

function getToolCallsFromParts({
  parts,
  generateId,
  useToolCallId = false,
}: {
  parts: ContentSchema['parts'];
  generateId: () => string;
  useToolCallId?: boolean;
}) {
  const functionCallParts = parts?.filter(
    part => 'functionCall' in part,
  ) as Array<
    GoogleGenerativeAIContentPart & {
      functionCall: { name: string; args: unknown; id?: string };
      thoughtSignature?: string | null;
    }
  >;

  return functionCallParts == null || functionCallParts.length === 0
    ? undefined
    : functionCallParts.map(part => ({
        type: 'tool-call' as const,
        // Use provided id for Claude models, generate for others
        toolCallId:
          useToolCallId && part.functionCall.id
            ? part.functionCall.id
            : generateId(),
        toolName: part.functionCall.name,
        args: JSON.stringify(part.functionCall.args),
        providerMetadata: part.thoughtSignature
          ? { google: { thoughtSignature: part.thoughtSignature } }
          : undefined,
      }));
}

function extractSources({
  groundingMetadata,
  generateId,
}: {
  groundingMetadata: GroundingMetadataSchema | undefined | null;
  generateId: () => string;
}): undefined | LanguageModelV2Source[] {
  if (!groundingMetadata?.groundingChunks) {
    return undefined;
  }

  const sources: LanguageModelV2Source[] = [];

  for (const chunk of groundingMetadata.groundingChunks) {
    if (chunk.web != null) {
      // Handle web chunks as URL sources
      sources.push({
        type: 'source',
        sourceType: 'url',
        id: generateId(),
        url: chunk.web.uri,
        title: chunk.web.title ?? undefined,
      });
    } else if (chunk.retrievedContext != null) {
      // Handle retrievedContext chunks from RAG operations
      const uri = chunk.retrievedContext.uri;
      const fileSearchStore = chunk.retrievedContext.fileSearchStore;

      if (uri && (uri.startsWith('http://') || uri.startsWith('https://'))) {
        // Old format: Google Search with HTTP/HTTPS URL
        sources.push({
          type: 'source',
          sourceType: 'url',
          id: generateId(),
          url: uri,
          title: chunk.retrievedContext.title ?? undefined,
        });
      } else if (uri) {
        // Old format: Document with file path (gs://, etc.)
        const title = chunk.retrievedContext.title ?? 'Unknown Document';
        let mediaType = 'application/octet-stream';
        let filename: string | undefined = undefined;

        if (uri.endsWith('.pdf')) {
          mediaType = 'application/pdf';
          filename = uri.split('/').pop();
        } else if (uri.endsWith('.txt')) {
          mediaType = 'text/plain';
          filename = uri.split('/').pop();
        } else if (uri.endsWith('.docx')) {
          mediaType =
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
          filename = uri.split('/').pop();
        } else if (uri.endsWith('.doc')) {
          mediaType = 'application/msword';
          filename = uri.split('/').pop();
        } else if (uri.match(/\.(md|markdown)$/)) {
          mediaType = 'text/markdown';
          filename = uri.split('/').pop();
        } else {
          filename = uri.split('/').pop();
        }

        sources.push({
          type: 'source',
          sourceType: 'document',
          id: generateId(),
          mediaType,
          title,
          filename,
        });
      } else if (fileSearchStore) {
        // New format: File Search with fileSearchStore (no uri)
        const title = chunk.retrievedContext.title ?? 'Unknown Document';
        sources.push({
          type: 'source',
          sourceType: 'document',
          id: generateId(),
          mediaType: 'application/octet-stream',
          title,
          filename: fileSearchStore.split('/').pop(),
        });
      }
    } else if (chunk.maps != null) {
      if (chunk.maps.uri) {
        sources.push({
          type: 'source',
          sourceType: 'url',
          id: generateId(),
          url: chunk.maps.uri,
          title: chunk.maps.title ?? undefined,
        });
      }
    }
  }

  return sources.length > 0 ? sources : undefined;
}

export const getGroundingMetadataSchema = () =>
  z.object({
    webSearchQueries: z.array(z.string()).nullish(),
    retrievalQueries: z.array(z.string()).nullish(),
    searchEntryPoint: z.object({ renderedContent: z.string() }).nullish(),
    groundingChunks: z
      .array(
        z.object({
          web: z
            .object({ uri: z.string(), title: z.string().nullish() })
            .nullish(),
          retrievedContext: z
            .object({
              uri: z.string().nullish(),
              title: z.string().nullish(),
              text: z.string().nullish(),
              fileSearchStore: z.string().nullish(),
            })
            .nullish(),
          maps: z
            .object({
              uri: z.string().nullish(),
              title: z.string().nullish(),
              text: z.string().nullish(),
              placeId: z.string().nullish(),
            })
            .nullish(),
        }),
      )
      .nullish(),
    groundingSupports: z
      .array(
        z.object({
          segment: z.object({
            startIndex: z.number().nullish(),
            endIndex: z.number().nullish(),
            text: z.string().nullish(),
          }),
          segment_text: z.string().nullish(),
          groundingChunkIndices: z.array(z.number()).nullish(),
          supportChunkIndices: z.array(z.number()).nullish(),
          confidenceScores: z.array(z.number()).nullish(),
          confidenceScore: z.array(z.number()).nullish(),
        }),
      )
      .nullish(),
    retrievalMetadata: z
      .union([
        z.object({
          webDynamicRetrievalScore: z.number(),
        }),
        z.object({}),
      ])
      .nullish(),
  });

const getContentSchema = () =>
  z.object({
    parts: z
      .array(
        z.union([
          // note: order matters since text can be fully empty
          z.object({
            functionCall: z.object({
              name: z.string(),
              args: z.unknown(),
              id: z.string().nullish(), // Claude models include tool call ID
            }),
            thoughtSignature: z.string().nullish(),
          }),
          z.object({
            inlineData: z.object({
              mimeType: z.string(),
              data: z.string(),
            }),
          }),
          z.object({
            executableCode: z
              .object({
                language: z.string(),
                code: z.string(),
              })
              .nullish(),
            codeExecutionResult: z
              .object({
                outcome: z.string(),
                output: z.string(),
              })
              .nullish(),
            text: z.string().nullish(),
            thought: z.boolean().nullish(),
            thoughtSignature: z.string().nullish(),
          }),
        ]),
      )
      .nullish(),
  });

// https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-filters
const getSafetyRatingSchema = () =>
  z.object({
    category: z.string().nullish(),
    probability: z.string().nullish(),
    probabilityScore: z.number().nullish(),
    severity: z.string().nullish(),
    severityScore: z.number().nullish(),
    blocked: z.boolean().nullish(),
  });

const usageSchema = z.object({
  cachedContentTokenCount: z.number().nullish(),
  thoughtsTokenCount: z.number().nullish(),
  promptTokenCount: z.number().nullish(),
  candidatesTokenCount: z.number().nullish(),
  totalTokenCount: z.number().nullish(),
  // https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse#TrafficType
  trafficType: z.string().nullish(),
});

// https://ai.google.dev/api/generate-content#UrlRetrievalMetadata
export const getUrlContextMetadataSchema = () =>
  z.object({
    urlMetadata: z.array(
      z.object({
        retrievedUrl: z.string(),
        urlRetrievalStatus: z.string(),
      }),
    ),
  });

const responseSchema = lazySchema(() =>
  zodSchema(
    z.object({
      candidates: z.array(
        z.object({
          content: getContentSchema().nullish().or(z.object({}).strict()),
          finishReason: z.string().nullish(),
          safetyRatings: z.array(getSafetyRatingSchema()).nullish(),
          groundingMetadata: getGroundingMetadataSchema().nullish(),
          urlContextMetadata: getUrlContextMetadataSchema().nullish(),
        }),
      ),
      usageMetadata: usageSchema.nullish(),
      promptFeedback: z
        .object({
          blockReason: z.string().nullish(),
          safetyRatings: z.array(getSafetyRatingSchema()).nullish(),
        })
        .nullish(),
    }),
  ),
);

type ContentSchema = NonNullable<
  InferValidator<typeof responseSchema>['candidates'][number]['content']
>;
export type GroundingMetadataSchema = NonNullable<
  InferValidator<
    typeof responseSchema
  >['candidates'][number]['groundingMetadata']
>;

type GroundingChunkSchema = NonNullable<
  GroundingMetadataSchema['groundingChunks']
>[number];

export type UrlContextMetadataSchema = NonNullable<
  InferValidator<
    typeof responseSchema
  >['candidates'][number]['urlContextMetadata']
>;

export type SafetyRatingSchema = NonNullable<
  InferValidator<typeof responseSchema>['candidates'][number]['safetyRatings']
>[number];

// Schema for the inner response content (used by both wrapped and unwrapped formats)
const getResponseContentSchema = () =>
  z.object({
    candidates: z
      .array(
        z.object({
          content: getContentSchema().nullish(),
          finishReason: z.string().nullish(),
          safetyRatings: z.array(getSafetyRatingSchema()).nullish(),
          groundingMetadata: getGroundingMetadataSchema().nullish(),
          urlContextMetadata: getUrlContextMetadataSchema().nullish(),
        }),
      )
      .nullish(),
    usageMetadata: usageSchema.nullish(),
    promptFeedback: z
      .object({
        blockReason: z.string().nullish(),
        safetyRatings: z.array(getSafetyRatingSchema()).nullish(),
      })
      .nullish(),
    modelVersion: z.string().nullish(),
    responseId: z.string().nullish(),
  });

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
// Cloud Assist wraps the response in a "response" field, but we handle both formats
const chunkSchema = lazySchema(() =>
  zodSchema(
    z.object({
      // Cloud Assist wrapper fields
      response: getResponseContentSchema().nullish(),
      traceId: z.string().nullish(),
      // Also allow direct fields for compatibility
      candidates: z
        .array(
          z.object({
            content: getContentSchema().nullish(),
            finishReason: z.string().nullish(),
            safetyRatings: z.array(getSafetyRatingSchema()).nullish(),
            groundingMetadata: getGroundingMetadataSchema().nullish(),
            urlContextMetadata: getUrlContextMetadataSchema().nullish(),
          }),
        )
        .nullish(),
      usageMetadata: usageSchema.nullish(),
      promptFeedback: z
        .object({
          blockReason: z.string().nullish(),
          safetyRatings: z.array(getSafetyRatingSchema()).nullish(),
        })
        .nullish(),
    }),
  ),
);

type ChunkSchema = InferValidator<typeof chunkSchema>;
