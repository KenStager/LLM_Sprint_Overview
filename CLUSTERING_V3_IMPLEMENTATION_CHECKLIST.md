# Clustering V3 Implementation Checklist

## Immediate Actions (Today)

### 1. ⚠️ Fix Critical Weight Bug
```bash
# Edit clustering/config/clustering_config.py
# Change weights to sum to 1.0:
perp_name: 0.45  # from 0.50
crime_location: 0.20  # from 0.30
text_similarity: 0.05  # from 0.10
```

### 2. 🚀 Run Clustering on 169 Articles
```bash
cd /Users/kenny/Desktop/Snapped_Production/
docker-compose -f docker-compose.production.yml run clustering python -m clustering
./run_duplicate_monitor.sh  # Verify no duplicates
```

### 3. 📊 Document Results
- Update cluster count
- Note any issues
- Check enrichment candidates

## Week 1 Implementation

### Day 1-2: Core Fixes
- [ ] Fix weight normalization bug
- [ ] Create database migration for geographic fields
- [ ] Implement Phase 2 clustering logic
- [ ] Test fixes locally

### Day 3-4: Unified Service
- [ ] Create `clustering/services/unified_clustering_service.py`
- [ ] Implement enrichment queue
- [ ] Add cost tracking
- [ ] Create format adapter

### Day 5: Testing
- [ ] Unit tests for unified service
- [ ] Integration test with both engines
- [ ] Verify format compatibility

## Week 2: Integration

### Day 1-2: Supporting Modules
- [ ] Implement `QualityAssessor`
- [ ] Create `MetricsCollector`
- [ ] Add monitoring hooks
- [ ] Set up Prometheus metrics

### Day 3-4: Testing Suite
- [ ] Write comprehensive unit tests
- [ ] Create performance benchmarks
- [ ] Add data quality tests
- [ ] Document test procedures

### Day 5: Documentation
- [ ] Update all READMEs
- [ ] Create API documentation
- [ ] Write deployment guide
- [ ] Update runbooks

## Code Cleanup Tasks

### Remove Redundant Files
```bash
# Backup first
tar -czf clustering_backup_$(date +%Y%m%d).tar.gz clustering/

# Remove duplicates
rm clustering/working_service.py.backup
rm clustering/working_service_backup.py
rm clustering/test_*.py  # One-off test scripts
```

### Reorganize Structure
- [ ] Move `structured_outputs/` → `enrichment/`
- [ ] Consolidate scripts in `scripts/`
- [ ] Create `services/` directory
- [ ] Add `monitoring/` directory

### Code Quality
- [ ] Add type hints to all functions
- [ ] Write docstrings for public methods
- [ ] Implement proper error handling
- [ ] Add structured logging

## Monitoring Setup

### Metrics to Track
- [ ] Clustering throughput (articles/sec)
- [ ] API costs ($/day)
- [ ] Cluster quality scores
- [ ] Error rates
- [ ] Queue sizes

### Alerts to Configure
- [ ] High error rate (>5%)
- [ ] API budget exceeded ($10/day)
- [ ] Low clustering throughput (<10/sec)
- [ ] Large backlog (>1000 articles)

## Success Criteria

### Technical
- ✅ Zero duplicate clusters
- ✅ Weights sum to 1.0
- ✅ All tests passing
- ✅ <2 sec clustering latency

### Operational  
- ✅ Daily runbooks in use
- ✅ Monitoring dashboard live
- ✅ Alert rules configured
- ✅ Team trained

### Business
- ✅ 95%+ cluster accuracy
- ✅ <$5/day API costs
- ✅ Rich case summaries
- ✅ Audit trail complete

## Questions to Answer

Before proceeding:
1. Do we have API keys for OpenAI configured?
2. Is the Prometheus endpoint available?
3. Are geographic fields critical for launch?
4. What is the daily article volume expected?

## Point of Contact

For questions or issues:
- Review: `CLUSTERING_ENGINE_STRATEGY_V3.md`
- Quick fixes: `CLUSTERING_V3_QUICK_REF.md`
- Operations: See runbooks in V3 strategy
