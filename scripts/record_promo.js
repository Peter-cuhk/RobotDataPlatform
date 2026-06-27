const fs = require("fs");
const path = require("path");
const { chromium } = require("/tmp/rds-video-tools/node_modules/playwright");

const root = process.cwd();
const outputDir = path.join(root, "output", "promo-video");
const videoDir = path.join(outputDir, "browser-video");
const screenshotDir = path.join(outputDir, "screens");
const datasetPath = "/Users/peterxie/Desktop/data platform /data/samples/aloha_static_coffee";
const appUrl = "http://127.0.0.1:5173/";

fs.mkdirSync(videoDir, { recursive: true });
fs.mkdirSync(screenshotDir, { recursive: true });

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function capture(page, name) {
  await page.screenshot({ path: path.join(screenshotDir, `${name}.png`), fullPage: false });
}

async function moveMouse(page, points, delay = 260) {
  for (const [x, y] of points) {
    await page.mouse.move(x, y, { steps: 16 });
    await sleep(delay);
  }
}

async function clickText(page, text, options = {}) {
  await page.getByText(text, options).first().click();
}

(async () => {
  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
    recordVideo: {
      dir: videoDir,
      size: { width: 1440, height: 900 },
    },
  });
  const page = await context.newPage();
  page.setDefaultTimeout(60000);

  await page.goto(appUrl, { waitUntil: "networkidle" });
  await sleep(900);
  await capture(page, "01-open");

  await page.getByRole("textbox", { name: "Dataset path" }).fill(datasetPath);
  await sleep(350);
  await capture(page, "02-path");
  await page.getByRole("button", { name: "Import dataset" }).click();
  await page.getByText("50 episodes").waitFor();
  await sleep(1300);
  await capture(page, "03-imported");

  await moveMouse(page, [
    [1110, 180],
    [960, 230],
    [1200, 305],
    [1040, 388],
  ]);

  const cleaningResponse = page.waitForResponse(
    (response) => response.url().includes("/cleaning/runs") && response.status() === 201,
    { timeout: 180000 },
  );
  const filtersResponse = page.waitForResponse(
    (response) => response.url().includes("/filters/runs") && response.status() === 201,
    { timeout: 180000 },
  );
  await page.getByRole("button", { name: "Run cleaning Pipeline" }).click();
  await cleaningResponse;
  await filtersResponse;
  await page.locator(".cleaning-summary-view").waitFor({ timeout: 30000 });
  await sleep(1800);
  await capture(page, "04-cleaned");

  await moveMouse(page, [
    [1120, 560],
    [1260, 665],
    [1020, 760],
    [520, 420],
  ]);

  await page.getByRole("button", { name: /^Sudden change$/ }).click();
  await page.locator(".filter-detail-page").waitFor({ timeout: 30000 });
  await sleep(1500);
  await capture(page, "05-filter-detail");

  await page.getByRole("button", { name: /^Episode 000010/ }).click();
  await sleep(900);
  await capture(page, "06-episode-switch");

  const recordingResponse = page.waitForResponse(
    (response) => response.url().includes("/recording") && response.status() === 201,
    { timeout: 120000 },
  );
  await page.getByRole("button", { name: "Replay in Rerun" }).click();
  await recordingResponse;
  await sleep(4500);
  await capture(page, "07-rerun");

  await page.getByRole("combobox", { name: "Export scope" }).selectOption("status_passed");
  await sleep(500);
  const exportResponse = page.waitForResponse(
    (response) => response.url().includes("/exports") && response.status() === 201,
    { timeout: 120000 },
  );
  await page.getByRole("button", { name: /^Export / }).click();
  await exportResponse;
  await page.getByText("Exported").waitFor({ timeout: 30000 });
  await sleep(1800);
  await capture(page, "08-exported");

  await moveMouse(page, [
    [1110, 215],
    [1110, 350],
    [1110, 510],
    [1140, 720],
  ]);
  await sleep(1000);

  const video = page.video();
  await context.close();
  await browser.close();
  const videoPath = await video.path();
  const targetPath = path.join(outputDir, "robot-data-studio-demo.webm");
  fs.copyFileSync(videoPath, targetPath);
  console.log(targetPath);
})();
