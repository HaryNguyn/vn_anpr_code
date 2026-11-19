# Sao chép thành .env.ps1 và điền giá trị thật. KHÔNG commit file .env.ps1.
$env:ANPR_API_KEY        = 'thay-bang-khoa-ngau-nhien'
$env:ANPR_MODEL_VERSION  = 'v1.0'
$env:ANPR_MAX_CONCURRENCY = '2'
$env:PARKING_WEBHOOK_URL = 'http://parking-backend.local/api/anpr/events'
$env:PARKING_API_KEY     = 'thay-bang-khoa-backend'
$env:CAM_USER            = 'thay'
$env:CAM_PASS            = 'thay'
