import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

export default withMermaid(
  defineConfig({
    title: "Ascent Interpreter",
    description: "Interpreter for Ascent Datalog programs",
    themeConfig: {
      nav: [
        { text: "Guide", link: "/guide/getting-started" },
        { text: "Reference", link: "/reference/syntax" },
      ],
      sidebar: [
        {
          text: "Guide",
          items: [
            { text: "Getting Started", link: "/guide/getting-started" },
          ],
        },
        {
          text: "Reference",
          items: [
            { text: "Syntax", link: "/reference/syntax" },
            { text: "Types & Aggregators", link: "/reference/types" },
            { text: "Architecture", link: "/reference/architecture" },
          ],
        },
      ],
      socialLinks: [
        {
          icon: "github",
          link: "https://github.com/s-arash/ascent",
        },
      ],
    },
  }),
);
