# bat-check.ps1 - verify (and optionally fix) the load-bearing byte-level
# properties of the Windows launchers: CRLF line endings, no UTF-8 BOM, ASCII only.
# cmd.exe mis-parses LF-only batch files and a BOM disables @echo off.
#   pwsh -File bat-check.ps1 -Path scripts\*.bat
#   pwsh -File bat-check.ps1 -Path scripts\*.bat -Fix
param(
  [Parameter(Mandatory = $true)][string[]]$Path,
  [switch]$Fix
)
$enc = [System.Text.Encoding]::GetEncoding(28591)
$bad = 0
foreach ($p in $Path) {
  $full = (Resolve-Path $p).Path
  $bytes = [System.IO.File]::ReadAllBytes($full)
  $bom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
  $text = $enc.GetString($bytes)
  $crlf = ([regex]::Matches($text, "`r`n")).Count
  $bareLf = ([regex]::Matches($text, "(?<!`r)`n")).Count
  $bareCr = ([regex]::Matches($text, "`r(?!`n)")).Count
  $nonAscii = 0
  foreach ($b in $bytes) { if ($b -gt 127) { $nonAscii++ } }

  if ($Fix) {
    $fixed = $text -replace "`r`n", "`n" -replace "`r", "`n" -replace "`n", "`r`n"
    [System.IO.File]::WriteAllBytes($full, $enc.GetBytes($fixed))
    $bytes = [System.IO.File]::ReadAllBytes($full)
    $text = $enc.GetString($bytes)
    $bom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
    $crlf = ([regex]::Matches($text, "`r`n")).Count
    $bareLf = ([regex]::Matches($text, "(?<!`r)`n")).Count
    $bareCr = ([regex]::Matches($text, "`r(?!`n)")).Count
    $nonAscii = 0
    foreach ($b in $bytes) { if ($b -gt 127) { $nonAscii++ } }
  }

  $ok = (-not $bom) -and ($bareLf -eq 0) -and ($bareCr -eq 0) -and ($nonAscii -eq 0) -and ($crlf -gt 0)
  if (-not $ok) { $bad++ }
  "{0,-12} {1,-34} lines(CRLF)={2,-5} bareLF={3,-4} bareCR={4,-4} BOM={5,-6} nonASCII={6}" -f `
    $(if ($ok) { 'OK' } else { 'FAIL' }), (Split-Path $full -Leaf), $crlf, $bareLf, $bareCr, $bom, $nonAscii
}
if ($bad -gt 0) { Write-Host "`n$bad file(s) violate the .bat byte contract"; exit 1 }
Write-Host "`nall files satisfy the .bat byte contract"