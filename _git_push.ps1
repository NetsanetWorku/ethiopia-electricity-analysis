$git = "C:\Program Files\Git\bin\git.exe"
$remote = "https://github.com/NetsanetWorku/ethiopia-electricity-analysis.git"
$cwd = "e:\Intership\Intership_2026_EEU\Ethiopian_EU\ethiopia-electricity-analysis"

Set-Location $cwd

Write-Host "==> git init"
& $git init

Write-Host "==> git config user"
& $git config user.name "NetsanetWorku"
& $git config user.email "netsanet@madwalabu.edu.et"

Write-Host "==> git remote add"
$remoteExists = & $git remote get-url origin 2>$null
if (-not $remoteExists) {
    & $git remote add origin $remote
} else {
    & $git remote set-url origin $remote
}

Write-Host "==> git add"
& $git add .

Write-Host "==> git status"
& $git status

Write-Host "==> git commit"
& $git commit -m "Initial commit: Ethiopia electricity analysis pipeline, Streamlit dashboard, and tests"

Write-Host "==> git branch -M main"
& $git branch -M main

Write-Host "==> git push"
& $git push -u origin main

Write-Host "==> Done"
