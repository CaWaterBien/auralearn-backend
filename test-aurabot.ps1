Write-Host "Testing Enhanced AuraBot" -ForegroundColor Yellow
Write-Host "========================" -ForegroundColor Yellow
Write-Host ""

Write-Host "Test 1: Simple greeting" -ForegroundColor Cyan
$body1 = @{
    question = "hi"
    html_context = "<!DOCTYPE html><html><head><title>Test</title></head><body><!-- comment --></body></html>"
    instructions_context = $null
    session_id = "test123"
    user_id = $null
} | ConvertTo-Json

try {
    $response1 = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/ask' -Method POST -ContentType 'application/json' -Body $body1
    Write-Host "SUCCESS: Greeting Response:" -ForegroundColor Green
    Write-Host $response1.response -ForegroundColor White
} catch {
    Write-Host "ERROR:" $_.Exception.Message -ForegroundColor Red
}

Write-Host ""
Write-Host "Test 2: Code request (should refuse)" -ForegroundColor Cyan
$body2 = @{
    question = "can you provide me the code?"
    html_context = "<!DOCTYPE html><html><head><title>Test</title></head><body><!-- comment --></body></html>"
    instructions_context = $null
    session_id = "test124"
    user_id = $null
} | ConvertTo-Json

try {
    $response2 = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/ask' -Method POST -ContentType 'application/json' -Body $body2
    Write-Host "SUCCESS: Code Request Response:" -ForegroundColor Green
    Write-Host $response2.response -ForegroundColor White
} catch {
    Write-Host "ERROR:" $_.Exception.Message -ForegroundColor Red
}

Write-Host ""
Write-Host "Test 3: Identity question" -ForegroundColor Cyan
$body3 = @{
    question = "who are you?"
    html_context = "<!DOCTYPE html><html><head><title>Test</title></head><body><!-- comment --></body></html>"
    instructions_context = $null
    session_id = "test125"
    user_id = $null
} | ConvertTo-Json

try {
    $response3 = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/ask' -Method POST -ContentType 'application/json' -Body $body3
    Write-Host "SUCCESS: Identity Response:" -ForegroundColor Green
    Write-Host $response3.response -ForegroundColor White
} catch {
    Write-Host "ERROR:" $_.Exception.Message -ForegroundColor Red
}

Write-Host ""
Write-Host "========================" -ForegroundColor Yellow
Write-Host "Test Complete!" -ForegroundColor Yellow
Read-Host "Press Enter to continue"
