FROM oven/bun:1 AS build

WORKDIR /app
COPY packages/web/package.json packages/web/bun.lock* ./
RUN bun install --frozen-lockfile

COPY packages/web/ .
RUN bun --bun run build

FROM oven/bun:1
WORKDIR /app
COPY --from=build /app/.output .output
CMD ["bun", "run", ".output/server/index.mjs"]
