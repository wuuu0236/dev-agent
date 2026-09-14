/**
 * build.mjs —— 把 React 应用打包成单个自包含 index.html
 *
 * 做了三件事：
 *   1. esbuild 把 app.jsx 连同 react / react-dom 打成单个 bundle.js（minify）
 *   2. 读取 template.html（页面骨架 + 全部 CSS，含 __BUNDLE__ 占位符）
 *   3. 把 bundle 注入占位符，产出 index.html —— 零外部依赖，双击可离线打开
 *
 * 用法：cd frontend && npm install && npm run build
 * 改完 app.jsx 后重跑 npm run build 即可。
 */
import { build } from "esbuild";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(fileURLToPath(import.meta.url));

const result = await build({
  entryPoints: [join(root, "app.jsx")],
  bundle: true,
  minify: true,
  write: false,
  charset: "utf8",
  define: { "process.env.NODE_ENV": '"production"' },
});

const js = result.outputFiles[0].text;
if (js.toLowerCase().includes("</script")) {
  throw new Error("bundle 中含有 </script>，直接内联会截断 HTML，需先转义");
}

const template = readFileSync(join(root, "template.html"), "utf-8");
const html = template.replace("__BUNDLE__", () => js);

writeFileSync(join(root, "index.html"), html, "utf-8");
console.log(`OK index.html ${(Buffer.byteLength(html) / 1024).toFixed(0)} KB（含 React，零外部依赖）`);
