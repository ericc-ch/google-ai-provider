import { describe, it, expect } from 'vitest';
import { getProviderKey } from './provider-key';

const ALLOWED_PROVIDER_KEYS = [
  'antigravity',
  'gemini-cli',
  'cloudassist',
  'google',
] as const;
type AllowedProviderKey = (typeof ALLOWED_PROVIDER_KEYS)[number];

describe('provider key detection', () => {
  it('should detect "google" key', () => {
    const providerOptions = { google: { foo: 'bar' } };
    expect(getProviderKey(providerOptions)).toBe('google');
  });

  it('should detect "cloudassist" key', () => {
    const providerOptions = { cloudassist: { foo: 'bar' } };
    expect(getProviderKey(providerOptions)).toBe('cloudassist');
  });

  it('should detect "antigravity" key', () => {
    const providerOptions = { antigravity: { foo: 'bar' } };
    expect(getProviderKey(providerOptions)).toBe('antigravity');
  });

  it('should detect "gemini-cli" key', () => {
    const providerOptions = { 'gemini-cli': { foo: 'bar' } };
    expect(getProviderKey(providerOptions)).toBe('gemini-cli');
  });

  it('should return undefined when no allowed key is present', () => {
    const providerOptions = { other: { foo: 'bar' } };
    expect(getProviderKey(providerOptions)).toBeUndefined();
  });

  it('should return undefined for empty provider options', () => {
    expect(getProviderKey({})).toBeUndefined();
  });

  it('should return undefined for undefined provider options', () => {
    expect(getProviderKey(undefined)).toBeUndefined();
  });

  it('should prefer the first key in ALLOWED_PROVIDER_KEYS that exists in providerOptions', () => {
    const providerOptions = {
      google: { foo: 'bar' },
      cloudassist: { baz: 'qux' },
    };
    expect(getProviderKey(providerOptions)).toBe('cloudassist');
  });
});
