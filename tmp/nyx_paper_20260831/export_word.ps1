param([string]$DocumentPath, [string]$PdfPath, [switch]$UpdateFields)
$ErrorActionPreference = 'Stop'
$cvWord = $null
$cvDocument = $null
function Invoke-CvCom {
    param([scriptblock]$Action)
    for ($cvAttempt = 0; $cvAttempt -lt 40; $cvAttempt++) {
        try { return (& $Action) }
        catch {
            if ($_.Exception.HResult -ne -2147418111 -and $_.Exception.Message -notmatch 'rejected by callee') { throw }
            Start-Sleep -Milliseconds 500
        }
    }
    throw 'Word remained busy during conversion.'
}
try {
    Write-Output 'Starting document conversion'
    $cvWord = New-Object -ComObject Word.Application
    $cvWord.Visible = $false
    $cvWord.DisplayAlerts = 0
    Write-Output 'Opening document'
    $cvDocument = $cvWord.Documents.Open($DocumentPath, $false, (-not $UpdateFields))
    Write-Output 'Rendering document'
    if ($UpdateFields) {
        Invoke-CvCom { $cvDocument.Fields.Update() } | Out-Null
        foreach ($cvToc in $cvDocument.TablesOfContents) { Invoke-CvCom { $cvToc.Update() } | Out-Null }
        Invoke-CvCom { $cvDocument.Repaginate() }
        foreach ($cvToc in $cvDocument.TablesOfContents) { Invoke-CvCom { $cvToc.UpdatePageNumbers() } | Out-Null }
        Invoke-CvCom { $cvDocument.Save() }
    }
    Invoke-CvCom { $cvDocument.Repaginate() }
    $cvPages = Invoke-CvCom { $cvDocument.ComputeStatistics(2) }
    Invoke-CvCom { $cvDocument.ExportAsFixedFormat($PdfPath, 17) }
    Write-Output "Pages: $cvPages"
    Write-Output "PDF: $PdfPath"
}
finally {
    if ($null -ne $cvDocument) {
        try { Invoke-CvCom { $cvDocument.Close(0) } } catch { Write-Warning $_ }
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($cvDocument)
    }
    if ($null -ne $cvWord) {
        try { Invoke-CvCom { $cvWord.Quit(0) } } catch { Write-Warning $_ }
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($cvWord)
    }
}
