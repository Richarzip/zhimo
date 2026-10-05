$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$assetDir = Join-Path $root 'frontend\static\assets\calligraphers'
$auditPath = Join-Path $root 'research\knowledge_audit.json'
New-Item -ItemType Directory -Force -Path $assetDir | Out-Null
$headers = @{ 'User-Agent' = 'ZhimoResearch/1.0 (local research)' }

$source = Get-Content (Join-Path $root 'src\zhimo\knowledge\data.py') -Raw -Encoding UTF8
$names = [regex]::Matches($source, '"name"\s*:\s*"([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
$entries = @()

function Get-Api($endpoint, $params) {
  $query = ($params.GetEnumerator() | ForEach-Object { "{0}={1}" -f $_.Key, [uri]::EscapeDataString([string]$_.Value) }) -join '&'
  Invoke-RestMethod -UseBasicParsing -Headers $headers -Uri ($endpoint + '?' + $query) -TimeoutSec 30
}

for($i = 0; $i -lt $names.Count; $i++) {
  $name = $names[$i]
  Write-Host ("[{0}/{1}] {2}" -f ($i + 1), $names.Count, $name)
  $wiki = @{ exact = $false; search = @() }
  try {
    $data = Get-Api 'https://zh.wikipedia.org/w/api.php' @{ action='query'; titles=$name; prop='extracts|info'; exintro=1; explaintext=1; inprop='url'; format='json'; utf8=1 }
    $page = @($data.query.pages.PSObject.Properties | ForEach-Object { $_.Value }) | Select-Object -First 1
    if($page -and -not $page.missing) { $wiki = @{ exact=$true; title=$page.title; pageid=$page.pageid; url=$page.fullurl; extract=$page.extract } }
  } catch { $wiki = @{ exact=$false; error='Wikipedia request failed' } }

  $images = @()
  try {
    $data = Get-Api 'https://commons.wikimedia.org/w/api.php' @{ action='query'; generator='search'; gsrsearch="$name filetype:bitmap"; gsrnamespace=6; gsrlimit=8; prop='imageinfo'; iiprop='url|mime|extmetadata'; iiurlwidth=1200; format='json'; utf8=1 }
    $pages = @($data.query.pages.PSObject.Properties | ForEach-Object { $_.Value })
    $n = 0
    foreach($page in $pages) {
      if($n -ge 2) { break }
      $info = @($page.imageinfo) | Select-Object -First 1
      if(-not $info -or $info.mime -notlike 'image/*' -or -not $info.thumburl) { continue }
      $n++
      $safe = ($name -replace '[^\p{L}\p{Nd}_-]', '_')
      $ext = if($info.mime -like '*jpeg*') { '.jpg' } elseif($info.mime -like '*png*') { '.png' } else { '.img' }
      $file = "${safe}_${n}${ext}"
      $path = Join-Path $assetDir $file
      try {
        Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri $info.thumburl -OutFile $path -TimeoutSec 60
        $meta = $info.extmetadata
        $images += [ordered]@{ title=$page.title; source_url=('https://commons.wikimedia.org/wiki/' + [uri]::EscapeDataString($page.title)); download_url=$info.thumburl; path=('frontend/static/assets/calligraphers/' + $file); url=('/static/assets/calligraphers/' + $file); license=if($meta.LicenseShortName){$meta.LicenseShortName.value}else{''} }
      } catch { }
    }
  } catch { }
  $entries += [ordered]@{ name=$name; wiki=$wiki; images=$images; review_status=if($wiki.exact){'source_found'}else{'needs_manual_review'}; review_notes=@('Manual review is still required for style claims and work attribution.') }
}
[ordered]@{ generated_at=(Get-Date).ToUniversalTime().ToString('o'); source_policy=@('https://zh.wikipedia.org/','https://commons.wikimedia.org/'); entries=$entries } | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $auditPath
Write-Host "Wrote $auditPath"
