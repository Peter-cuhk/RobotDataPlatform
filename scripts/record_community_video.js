const fs = require("fs");
const path = require("path");
const { chromium } = require("/tmp/rds-video-tools/node_modules/playwright");

const root = process.cwd();
const outputDir = path.join(root, "output", "community-video");
const rawDir = path.join(outputDir, "raw");
const stillDir = path.join(outputDir, "stills");
const datasetPath = path.join(root, "data", "samples", "aloha_static_coffee");
const appUrls = ["http://localhost:5173/", "http://127.0.0.1:5173/"];

fs.mkdirSync(rawDir, { recursive: true });
fs.mkdirSync(stillDir, { recursive: true });

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function capture(page, name) {
  await page.screenshot({ path: path.join(stillDir, `${name}.png`), fullPage: false });
}

async function gotoApp(page) {
  let lastError;
  for (const url of appUrls) {
    try {
      await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
      return url;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

async function clickFirstVisible(page, candidates) {
  for (const locator of candidates) {
    const count = await locator.count().catch(() => 0);
    for (let i = 0; i < count; i += 1) {
      const item = locator.nth(i);
      if (await item.isVisible().catch(() => false)) {
        await item.click();
        return true;
      }
    }
  }
  return false;
}

async function moveMouseTour(page, points, delay = 180) {
  for (const [x, y] of points) {
    await page.mouse.move(x, y, { steps: 18 });
    await sleep(delay);
  }
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
      dir: rawDir,
      size: { width: 1440, height: 900 },
    },
  });
  const page = await context.newPage();
  page.setDefaultTimeout(90000);

  const appUrl = await gotoApp(page);
  await sleep(900);
  await capture(page, "01-open");

  const pathInput = page.getByRole("textbox", { name: "Dataset path" });
  await pathInput.fill(datasetPath);
  await sleep(350);
  await capture(page, "02-path");

  await page.getByRole("button", { name: "Import dataset" }).click();
  await page.getByText(/episodes/i).first().waitFor({ timeout: 120000 });
  await page.getByText(/frames/i).first().waitFor({ timeout: 120000 });
  await sleep(1300);
  await capture(page, "03-imported");

  await moveMouseTour(page, [
    [1090, 178],
    [965, 228],
    [1200, 306],
    [1040, 388],
  ]);

  await page.getByRole("button", { name: /Run selected episode/i }).click();
  await page.locator(".cleaning-summary-view").waitFor({ timeout: 240000 });
  await sleep(1800);
  await capture(page, "04-cleaned");

  await moveMouseTour(page, [
    [1120, 560],
    [1260, 665],
    [1020, 760],
    [520, 420],
  ]);

  const detailOpened = await clickFirstVisible(page, [
    page.getByRole("button", { name: /^Visual quality$/i }),
    page.getByRole("button", { name: /^Sudden change$/i }),
    page.getByRole("button", { name: /^Time sync$/i }),
    page.getByText(/^Visual quality$/i),
    page.getByText(/^Sudden change$/i),
  ]);
  if (detailOpened) {
    await page.locator(".filter-detail-page").waitFor({ timeout: 60000 }).catch(() => null);
    await sleep(1500);
  }
  await capture(page, "05-filter-detail");

  await clickFirstVisible(page, [
    page.getByRole("button", { name: /^Episode 000010/i }),
    page.getByRole("button", { name: /^Episode 000030/i }),
    page.getByRole("button", { name: /^#000010/i }),
  ]);
  await sleep(900);
  await capture(page, "06-episode-focus");

  const recordingResponse = page.waitForResponse(
    (response) => response.url().includes("/recording") && response.status() === 201,
    { timeout: 180000 },
  );
  await page.getByRole("button", { name: "Replay in Rerun" }).click();
  await recordingResponse.catch(() => null);
  await sleep(6000);
  await capture(page, "07-rerun");

  await clickFirstVisible(page, [
    page.getByRole("combobox", { name: "Export scope" }),
    page.locator("select").filter({ hasText: /Selected episode|Passed episodes|All episodes/i }),
  ]);
  const exportScope = page.getByRole("combobox", { name: "Export scope" });
  if (await exportScope.count()) {
    await exportScope.selectOption("status_passed").catch(async () => {
      await exportScope.selectOption("selected");
    });
  }
  await sleep(500);

  const exportResponse = page.waitForResponse(
    (response) => response.url().includes("/exports") && response.status() === 201,
    { timeout: 180000 },
  );
  await page.getByRole("button", { name: /^Export / }).click();
  await exportResponse.catch(() => null);
  await page.getByText(/Exported/i).waitFor({ timeout: 60000 }).catch(() => null);
  await sleep(1800);
  await capture(page, "08-exported");

  await moveMouseTour(page, [
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
  const targetPath = path.join(rawDir, "workflow-full.webm");
  fs.copyFileSync(videoPath, targetPath);
  console.log(JSON.stringify({ appUrl, video: targetPath, stills: stillDir }, null, 2));
})();
