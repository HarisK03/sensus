import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import viteReact from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";

/** Subpath hosting (e.g. GitHub project pages): set BASE_PATH=/repo-name/ when building. */
const rawBase = process.env.BASE_PATH || "/";
const base = rawBase === "/" ? "/" : rawBase.endsWith("/") ? rawBase : `${rawBase}/`;
const routerBasepath = base === "/" ? "/" : base.replace(/\/$/, "");

export default defineConfig({
  base,
  plugins: [
    tailwindcss(),
    tsconfigPaths(),
    tanstackStart({
      srcDirectory: "src",
      server: { entry: "server" },
      router: { basepath: routerBasepath },
      prerender: {
        enabled: true,
        crawlLinks: true,
      },
    }),
    viteReact(),
  ],
});
