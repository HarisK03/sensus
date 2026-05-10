#!/bin/bash
set -e
# Node.js 20 via NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Playwright Firefox
pip install playwright --break-system-packages
playwright install firefox

# IBM Equal Access sidecar
sudo mkdir -p /opt/sensus/accessibility
sudo chown -R "$USER":"$USER" /opt/sensus/accessibility

cat > /opt/sensus/accessibility/package.json << 'EOF'
{
  "name": "sensus-accessibility",
  "version": "1.0.0",
  "dependencies": {
    "accessibility-checker": "^3.1.0"
  }
}
EOF
cd /opt/sensus/accessibility && npm install

# checker.js sidecar
cat > /opt/sensus/accessibility/checker.js << 'EOF'
const aChecker = require("accessibility-checker");
const { chromium } = require("playwright");

(async () => {
  const url = process.argv[2];
  if (!url) { console.error("No URL"); process.exit(1); }
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "networkidle" });
  const results = await aChecker.getCompliance(page, "page");
  const elements = (results.report?.results || []).map(r => ({
    ruleId: r.ruleId,
    level: r.level,
    path: r.path?.dom,
    message: r.message
  }));
  console.log(JSON.stringify(elements));
  await browser.close();
  process.exit(0);
})();
EOF

playwright install

echo "Browser setup complete."
