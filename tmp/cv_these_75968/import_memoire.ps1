$ErrorActionPreference = 'Stop'
$memoWord = $null
$memoDocument = $null
try {
    Write-Output 'Opening source PDF'
    $memoWord = New-Object -ComObject Word.Application
    $memoWord.Visible = $false
    $memoWord.DisplayAlerts = 0
    $memoDocument = $memoWord.Documents.Open('C:\Users\BQ6757\Downloads\Memoire_Yoan_Kesraoui.pdf', $false, $true)
    Write-Output 'Saving editable working document'
    $memoDocument.SaveAs2('C:\Users\BQ6757\chronos2_v1\tmp\cv_these_75968\memoire_reflow.docx', 16)
    Write-Output 'Conversion complete'
}
finally {
    if ($null -ne $memoDocument) {
        $memoDocument.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($memoDocument)
    }
    if ($null -ne $memoWord) {
        $memoWord.Quit(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($memoWord)
    }
}
