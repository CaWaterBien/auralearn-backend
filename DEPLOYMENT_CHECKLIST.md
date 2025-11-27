# 🚀 AuraBot RAG System - Deployment Checklist

## Pre-Deployment Requirements

### ✅ Environment Setup
- [ ] PostgreSQL database with pgvector extension enabled
- [ ] PHP 8.2+ with required extensions
- [ ] Composer installed and updated
- [ ] Node.js 18+ installed
- [ ] OpenAI API key obtained and configured
- [ ] Nebius API key configured (already provided)

### ✅ Code Deployment
- [ ] All files uploaded to server
- [ ] Composer dependencies installed: `composer install --optimize-autoloader --no-dev`
- [ ] Node dependencies installed: `npm install`
- [ ] Environment file configured: `.env` with all required variables
- [ ] File permissions set correctly

### ✅ Database Setup
- [ ] Database created and accessible
- [ ] pgvector extension enabled: `CREATE EXTENSION IF NOT EXISTS vector;`
- [ ] Migrations run: `php artisan migrate`
- [ ] Sample data created: `php artisan rag:create-samples`
- [ ] Course content ingested: `php artisan rag:ingest-courses`

### ✅ Configuration Verification
- [ ] API keys working: `php artisan rag:test`
- [ ] Nebius API accessible from server
- [ ] OpenAI API accessible for embeddings
- [ ] CORS configured for frontend domain
- [ ] Session storage working

### ✅ Performance Optimization
- [ ] Database indexes created (automatic with migrations)
- [ ] Application cache optimized: `php artisan optimize`
- [ ] Route cache cleared: `php artisan route:cache`
- [ ] Config cache cleared: `php artisan config:cache`
- [ ] View cache optimized: `php artisan view:cache`

### ✅ Security Configuration
- [ ] APP_DEBUG=false in production
- [ ] APP_ENV=production
- [ ] Strong APP_KEY generated
- [ ] CORS origins properly configured
- [ ] API rate limiting enabled
- [ ] Error pages customized (no sensitive data exposed)

### ✅ Frontend Integration
- [ ] Frontend built and deployed
- [ ] API_BASE environment variable points to backend
- [ ] CORS working between frontend and backend
- [ ] AuraBot integration tested end-to-end

## Production Environment Variables

### Required `.env` Configuration
```env
# Application
APP_NAME=AuraLearn
APP_ENV=production
APP_DEBUG=false
APP_URL=https://your-domain.com

# Database (PostgreSQL with pgvector)
DB_CONNECTION=pgsql
DB_HOST=your-postgres-host
DB_PORT=5432
DB_DATABASE=auralearn_production
DB_USERNAME=your-db-user
DB_PASSWORD=your-secure-password

# AI APIs
OPENAI_API_KEY=your_openai_api_key_here
NEBIUS_API_KEY=eyJhbGciOiJIUzI1NiIsImtpZCI6...  # Already configured
NEBIUS_BASE_URL=https://api.studio.nebius.com/v1/
NEBIUS_MODEL=openai/gpt-oss-20b

# RAG Configuration
VECTOR_DIM=1536
RAG_MAX_CHUNKS=5
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
AURABOT_MAX_TOKENS=5000
AURABOT_ATTEMPT_LIMIT=3

# Cache & Performance
CACHE_STORE=redis  # Or database
QUEUE_CONNECTION=redis  # For background processing
SESSION_DRIVER=database
```

## Health Checks

### Automated Verification
```bash
# Complete system test
php artisan rag:test

# Health endpoint
curl https://your-domain.com/api/aurabot/health

# Database connection
php artisan tinker
# In tinker: \App\Models\RagDocument::count()
```

### Expected Results
- ✅ Database: Connected with documents > 0
- ✅ Nebius API: Connected with test response
- ✅ Embeddings: Generated successfully
- ✅ RAG Search: Returns relevant documents
- ✅ AuraBot Workflow: Complete end-to-end success

## Monitoring & Maintenance

### Daily Checks
- [ ] Health endpoint responding: `/api/aurabot/health`
- [ ] Error logs reviewed: `storage/logs/laravel.log`
- [ ] API quota monitoring (OpenAI/Nebius)
- [ ] Database performance monitoring

### Weekly Maintenance
- [ ] Embedding cache cleanup: `php artisan rag:cleanup-cache`
- [ ] Database backup verification
- [ ] Performance metrics review
- [ ] User feedback analysis

### Monthly Tasks
- [ ] Content updates: Re-run `php artisan rag:ingest-courses`
- [ ] System updates: Update dependencies
- [ ] Security patches: Apply Laravel/PHP updates
- [ ] Analytics review: User engagement metrics

## Troubleshooting Production Issues

### Common Production Problems

#### 1. "AuraBot not responding"
```bash
# Check service health
curl https://your-domain.com/api/aurabot/health

# Check logs
tail -f storage/logs/laravel.log

# Restart queue workers
php artisan queue:restart
```

#### 2. "Slow response times"
```bash
# Check database performance
php artisan tinker
# In tinker: DB::connection()->enableQueryLog()

# Optimize application
php artisan optimize:clear
php artisan optimize
```

#### 3. "Memory/timeout issues"
```bash
# Increase PHP limits in php.ini
memory_limit = 512M
max_execution_time = 300

# Or use queue for heavy operations
php artisan queue:work
```

## Scaling Considerations

### Horizontal Scaling
- Use Redis for session storage
- Implement queue workers for embeddings
- Consider dedicated vector database (Pinecone, Weaviate)
- Load balance with sticky sessions

### Performance Optimization
- Implement response caching
- Use CDN for static assets
- Optimize database queries
- Consider embedding model alternatives

## Security Hardening

### Production Security
- [ ] HTTPS enforced
- [ ] API rate limiting enabled
- [ ] Input sanitization verified
- [ ] Error handling reviewed
- [ ] Logging configured (no sensitive data)
- [ ] Database access restricted
- [ ] API keys secured (not in logs)

## Backup & Recovery

### Critical Data
- [ ] Database (conversations, sessions, documents)
- [ ] Embedding cache (optional - can be regenerated)
- [ ] Configuration files
- [ ] SSL certificates

### Recovery Procedures
- [ ] Database restore process documented
- [ ] RAG content re-ingestion process
- [ ] Emergency contact information
- [ ] Rollback procedures documented

---

## 🎯 Deployment Success Criteria

Your deployment is successful when:

1. **Health Check Passes**: `/api/aurabot/health` returns green
2. **End-to-End Working**: Frontend → Backend → AI → Response
3. **Attempt Limiting Works**: 3 questions then block
4. **Context Detection Works**: HTML/instructions automatically detected
5. **Conversation Persists**: Chat history maintained across sessions
6. **Performance Acceptable**: < 5 second response times
7. **Error Handling Works**: Graceful degradation on failures

## 📞 Production Support

### Emergency Contacts
- System Administrator: [Your Contact]
- Database Administrator: [Your Contact]  
- AI/API Support: [Your Contact]

### Quick Fixes
```bash
# Restart all services
php artisan optimize:clear
php artisan queue:restart
systemctl restart php8.2-fpm  # Adjust for your PHP version
systemctl restart nginx       # Or apache2

# Reset problematic session
php artisan tinker
# In tinker: \App\Models\ChatbotSession::where('session_id', 'problem_session')->first()->resetAttempts()
```

---

**🎉 Your AuraBot RAG system is production-ready! Students can now get intelligent, educational assistance powered by cutting-edge AI technology.**

