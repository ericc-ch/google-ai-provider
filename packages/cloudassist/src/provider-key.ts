export const ALLOWED_PROVIDER_KEYS = [
  'antigravity',
  'gemini-cli',
  'cloudassist',
  'google',
] as const;
export type AllowedProviderKey = (typeof ALLOWED_PROVIDER_KEYS)[number];

export function getProviderKey(
  providerOptions: any,
): AllowedProviderKey | undefined {
  return ALLOWED_PROVIDER_KEYS.find(
    key => providerOptions?.[key] !== undefined,
  );
}
