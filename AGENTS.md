# AI SDK - Agent Guidelines

This repository is a monorepo containing the AI SDK and related packages, managed with TurboRepo and pnpm.

## Build, Lint, and Test

### Core Commands
- **Install Dependencies**: `pnpm install`
- **Build All**: `pnpm build` (Runs `turbo build --concurrency 16`)
- **Lint**: `pnpm lint`
- **Type Check**: `pnpm type-check` (Uses `tsc --build`)
- **Format Check**: `pnpm prettier-check`
- **Fix Formatting**: `pnpm prettier-fix`

### Testing
The project uses **Vitest** for testing.

- **Run All Tests**: `pnpm test` (skips examples)
- **Run Tests in Specific Package**: Navigate to the package directory (e.g., `cd packages/google`) and run `pnpm test`.
- **Run a Single Test File**: 
  ```bash
  # Inside a package directory
  pnpm test [path/to/test-file]
  # Example: pnpm test src/e2e/google.test.ts
  ```
- **Watch Mode**: `pnpm test --watch` (available in some packages)

### Workflow
1. **New Branch**: Create a new branch for your changes.
2. **Changesets**: If modifying packages (not examples), run `pnpm changeset` to generate a patch/minor/major version bump.
3. **Update References**: If adding package dependencies, run `pnpm update-references` in the root.

## Code Style & Conventions

### General
- **Language**: TypeScript (Strict mode enabled).
- **Formatting**: Enforced by Prettier.
  - Indentation: 2 spaces
  - Quotes: Single quotes (`'`)
  - Semicolons: Yes
  - Trailing Commas: All (ES5+)
- **Linter**: ESLint with `eslint-config-vercel-ai`.

### Imports
- Use named imports where possible.
- Group imports: external dependencies first, then internal/relative imports.
- Use absolute paths or aliases defined in `tsconfig.json` if available, otherwise relative paths.

### Naming
- **Variables/Functions**: `camelCase`
- **Classes/Interfaces/Types**: `PascalCase`
- **Files**: `kebab-case.ts` (mostly), or matching export name.
- **Constants**: `UPPER_CASE` for global constants, `camelCase` for others.

### Error Handling
- Use `async/await` for asynchronous operations.
- Handle errors gracefully using `try/catch` blocks where appropriate.
- Do not suppress errors without a valid reason and comment.

### Component/Function Structure
- Prefer functional components/composition.
- Keep functions small and focused.
- Add comments only for complex logic or "why" something is done; avoid obvious "what" comments.

## Project Structure
- `packages/`: Core libraries (e.g., `ai`, providers like `google`, `anthropic`).
- `examples/`: Usage examples (not published to npm).
- `content/`: Documentation.

## Rules
- **No Force Push**: Do not force push to shared branches.
- **Lockfile**: Keep `pnpm-lock.yaml` up to date (`pnpm install`).
- **Filesystem**: Always use absolute paths when using agent tools.
