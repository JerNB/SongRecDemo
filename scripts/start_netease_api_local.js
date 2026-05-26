const fs = require('fs')
const os = require('os')
const path = require('path')

const root = path.resolve(__dirname, '..')
const apiDir = path.join(root, 'vendor', 'api-enhanced')

async function main() {
  process.chdir(apiDir)

  const tokenPath = path.join(os.tmpdir(), 'anonymous_token')
  if (!fs.existsSync(tokenPath)) {
    fs.writeFileSync(tokenPath, '', 'utf-8')
  }

  await require(path.join(apiDir, 'generateConfig'))()
  await require(path.join(apiDir, 'server')).serveNcmApi({
    checkVersion: false,
  })
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
