# SNAPPED Clustering Continuation Prompt

## Context
I need to continue work on the SNAPPED Pipeline clustering system. In the previous session, we analyzed two clustering engines and identified critical bugs that need immediate fixes before we can proceed with implementation.

## System Information
- **Working Directory**: `/Users/kenny/Desktop/Snapped_Production/`
- **Database**: Supabase project `ahfwzfmetiqpvyzhrgxh` (MCP connected)
- **Docker**: Production setup ready
- **Current State**: 183 clusters (0 duplicates), 169 unclustered articles waiting

## Critical Issues Requiring Immediate Fix

### 1. Weight Configuration Bug
The similarity weights in `/clustering/config/clustering_config.py` sum to 1.2 instead of 1.0:
```python
# BROKEN - sums to 1.2:
perp_name: 0.50
crime_type: 0.15
crime_date: 0.15
crime_location: 0.30
text_similarity: 0.10
```

This causes inflated similarity scores and needs immediate correction to:
```python
# FIXED - sums to 1.0:
perp_name: 0.45
crime_type: 0.15
crime_date: 0.15
crime_location: 0.20
text_similarity: 0.05
```

### 2. Unclustered Articles
169 articles are ready for clustering. Once the weight bug is fixed, run:
```bash
docker-compose -f docker-compose.production.yml run clustering python -m clustering
./run_duplicate_monitor.sh
```

## Implementation Status

### What We Have
1. **ClusterEngine**: Currently active, prevents duplicates, no LLM usage
2. **HybridClusteringEngine**: Creates rich summaries with LLM, but lacks multi-session support
3. **Strategy Document**: CLUSTERING_ENGINE_STRATEGY_V3.md with complete UnifiedClusteringService implementation

### What We Need to Build
1. **UnifiedClusteringService**: Combines both engines (code provided in V3 strategy)
2. **FormatAdapter**: Converts between incompatible JSON formats
3. **Enrichment Queue**: Background processing for LLM enrichment
4. **Monitoring**: Metrics collection and cost tracking

## Immediate Tasks

### Phase 1: Fix and Test (Today)
1. Fix weight configuration bug
2. Run clustering on 169 articles
3. Verify no duplicates created
4. Identify high-priority clusters for enrichment

### Phase 2: Implementation (This Week)
1. Create `/clustering/services/unified_clustering_service.py` from V3 strategy
2. Implement FormatAdapter for JSON compatibility
3. Add enrichment queue with cost controls
4. Test with selective enrichment

### Phase 3: Cleanup (Next Week)
1. Remove backup files (working_service_backup.py, etc.)
2. Consolidate test files into tests/ directory
3. Add proper logging and monitoring
4. Update documentation

## Key Files to Reference
1. **Current Implementation**: `/clustering/working_service.py`
2. **Strategy**: `/CLUSTERING_ENGINE_STRATEGY_V3.md` (complete implementation)
3. **Config**: `/clustering/config/clustering_config.py` (needs weight fix)
4. **Monitoring**: `./run_duplicate_monitor.sh`

## Technical Details
- **ClusterEngine** uses multi-factor similarity with veto logic
- **HybridEngine** uses simple similarity but creates better summaries
- JSON formats are incompatible between engines
- UI expects HybridEngine format

## Success Criteria
- Weight bug fixed and clustering runs successfully
- ~80-100 new clusters created from 169 articles
- Zero duplicates (verified by monitor)
- UnifiedClusteringService implemented and tested
- Selective enrichment working with <$1/day API cost

## Questions to Resolve
1. Should we add geographic fields to database now or defer?
2. What's the priority threshold for cluster enrichment?
3. Do we have OpenAI API keys configured for enrichment?

Please help me:
1. First, fix the critical weight configuration bug
2. Run clustering on the 169 unclustered articles
3. Begin implementing the UnifiedClusteringService from the V3 strategy
4. Avoid creating new workaround code - fix issues directly in the existing codebase

All implementation code is provided in CLUSTERING_ENGINE_STRATEGY_V3.md - we just need to execute the plan.
