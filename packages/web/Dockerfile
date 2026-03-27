FROM oven/bun:1 AS build

WORKDIR /app
COPY packages/web/package.json packages/web/bun.lock* ./
RUN bun install --frozen-lockfile

COPY packages/web/ .
RUN bun --bun run build

FROM oven/bun:1
WORKDIR /app
COPY packages/web/package.json packages/web/bun.lock* ./
RUN bun install --frozen-lockfile --production
COPY --from=build /app/dist ./dist
CMD ["bun", "dist/server/server.js"]
