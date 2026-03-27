/// <reference types="vite/client" />
import { createRootRoute, HeadContent, Outlet, Scripts } from '@tanstack/react-router';
import type { PublicRuntimeConfig } from '@/lib/runtime-config';
import appCss from '@/styles/app.css?url';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'Triton Playground' },
      {
        name: 'description',
        content: 'Speech model demo with Triton Inference Server.',
      },
    ],
    links: [{ rel: 'stylesheet', href: appCss }],
  }),
  component: RootComponent,
  shellComponent: RootDocument,
});

function RootDocument({ children }: { children: React.ReactNode }) {
  const runtimeConfig: PublicRuntimeConfig = {
    workerApiUrl: process.env.WORKER_API_URL ?? process.env.VITE_WORKER_API_URL ?? null,
    workerPort: process.env.WORKER_PORT ?? '8080',
    webPort: process.env.WEB_PORT ?? '4000',
  };

  const runtimeConfigScript = `window.__TRITON_PLAYGROUND_CONFIG__=${JSON.stringify(runtimeConfig).replace(/</g, '\\u003c')};`;

  return (
    <html lang='en'>
      <head>
        <HeadContent />
      </head>
      <body className='min-h-screen bg-background text-foreground antialiased'>
        {children}
        <script dangerouslySetInnerHTML={{ __html: runtimeConfigScript }} />
        <Scripts />
      </body>
    </html>
  );
}

function RootComponent() {
  return <Outlet />;
}
