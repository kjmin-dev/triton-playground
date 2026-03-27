import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import { defineConfig, loadEnv } from "vite";
import viteReact from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

const ROOT_ENV_DIR = path.resolve(__dirname, "../..");

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ROOT_ENV_DIR, "");
  const webHost = env.WEB_HOST?.trim() || "0.0.0.0";
  const webPort = Number.parseInt(env.WEB_PORT || "4000", 10);

  return {
    envDir: ROOT_ENV_DIR,
    envPrefix: ["VITE_", "WORKER_", "WEB_"],
    server: {
      host: webHost,
      port: Number.isFinite(webPort) ? webPort : 4000,
    },
    resolve: {
      tsconfigPaths: true,
    },
    plugins: [tailwindcss(), tanstackStart({ srcDirectory: "src" }), viteReact()],
  };
});
