const fs = require("fs");
const path = require("path");
const { chromium } = require("/tmp/rds-video-tools/node_modules/playwright");

const root = process.cwd();
const outputDir = path.join(root, "output", "promo-video");
const videoDir = path.join(outputDir, "tail-video");
const screenshotDir = path.join(outputDir, "screens");
const datasetPath = "/Users/peterxie/Desktop/data platform /data/samples/aloha_static_coffee";

fs.mkdirSync(videoDir, { recursive: true });
fs.mkdirSync(screenshotDir, { recursive: true });

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function capture(page, name) {
  await page.screenshot({ path: path.join(screenshotDir, `${name}.png`), fullPage: false });
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
    recordVideo: {
      dir: videoDir,
      size: { width: 1440, height: 900 },
    },
  });
  const page = await context.newPage();
  page.setDefaultTimeout(90000);

  await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });
  await page.getByRole("textbox", { name: "Dataset path" }).fill(datasetPath);
  await page.getByRole("button", { name: "Import dataset" }).click();
  await page.getByText("50 episodes").waitFor();
  await sleep(1000);
  await capture(page, "06-tail-imported");

  const recordingResponse = page.waitForResponse(
    (response) => response.url().includes("/recording") && response.status() === 201,
    { timeout: 120000 },
  );
  await page.getByRole("button", { name: "Replay in Rerun" }).click();
  await recordingResponse;
  await sleep(5000);
  await capture(page, "07-rerun");

  const exportResponse = page.waitForResponse(
    (response) => response.url().includes("/exports") && response.status() === 201,
    { timeout: 120000 },
  );
  await page.getByRole("button", { name: /^Export / }).click();
  await exportResponse;
  await page.getByText("Exported").waitFor();
  await sleep(1500);
  await capture(page, "08-exported");

  await context.close();
  await browser.close();
  console.log(path.join(videoDir, "recorded tail"));
})();
