# make-image.ps1 - generate the vision probe image for the SparkInfer ctx-163840 run.
# Content is verifiable: a clearance code (E-7741) and three shapes with known colors.
Add-Type -AssemblyName System.Drawing

$out = 'D:\Code\MJ-Project\ai-model-nvfp4\.scratch\sparkinfer-ctx163840\image-text.png'
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null

$bmp = New-Object System.Drawing.Bitmap 1024, 640
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.Clear([System.Drawing.Color]::White)

$black = [System.Drawing.Brushes]::Black
$fTitle = New-Object System.Drawing.Font 'Arial', 40, ([System.Drawing.FontStyle]::Bold)
$fCode = New-Object System.Drawing.Font 'Arial', 56, ([System.Drawing.FontStyle]::Bold)
$fSmall = New-Object System.Drawing.Font 'Arial', 28, ([System.Drawing.FontStyle]::Regular)

$g.DrawString('AURORA-9 CLEARANCE', $fTitle, $black, 40, 30)
$g.DrawString('CODE: E-7741', $fCode, $black, 40, 100)
$g.DrawString('issued 2026-09-18 / station 42', $fSmall, $black, 40, 195)

$g.FillEllipse([System.Drawing.Brushes]::Red, 120, 320, 170, 170)
$g.FillRectangle([System.Drawing.Brushes]::Green, 420, 320, 170, 170)
$tri = [System.Drawing.Point[]]@(
  [System.Drawing.Point]::new(640, 490),
  [System.Drawing.Point]::new(810, 490),
  [System.Drawing.Point]::new(725, 320)
)
$g.FillPolygon([System.Drawing.Brushes]::Blue, $tri)

$g.DrawString('shape check: circle / square / triangle', $fSmall, $black, 120, 540)

$g.Dispose()
$bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()

Get-Item $out | Select-Object FullName, Length