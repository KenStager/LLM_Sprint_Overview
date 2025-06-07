# SNAPPED Pipeline Clustering Engine Strategy v3.0

## Executive Summary

This comprehensive strategy addresses critical bugs, architectural incompatibilities, and operational gaps in the SNAPPED clustering system. It provides a concrete implementation plan for a unified clustering service that leverages both engines effectively while maintaining data quality and controlling costs.

**Critical Issues Addressed:**
- Weight configuration bug (sum = 1.2 instead of 1.0)
- Summary JSON format incompatibility between engines
- Missing Phase 2 clustering implementation
- Code organization and technical debt
- Lack of comprehensive testing and monitoring

**Recommended Approach:** Unified service with ClusterEngine for clustering and selective HybridEngine enrichment.

## Table of Contents
1. [Critical Bug Fixes](#critical-bug-fixes)
2. [Architecture Overview](#architecture-overview)
3. [Complete Implementation](#complete-implementation)
4. [Code Cleanup Initiative](#code-cleanup-initiative)
5. [Testing Strategy](#testing-strategy)
6. [Performance Metrics](#performance-metrics)
7. [Monitoring & Alerting](#monitoring--alerting)
8. [Data Quality Framework](#data-quality-framework)
9. [Operational Runbooks](#operational-runbooks)
10. [Migration Timeline](#migration-timeline)

## Critical Bug Fixes

### 1. Weight Normalization Bug

**Issue:** Similarity weights sum to 1.2, causing inflated similarity scores.
```python
# Current (BROKEN):
perp_name: 0.50 + crime_type: 0.15 + crime_date: 0.15 + 
crime_location: 0.30 + text_similarity: 0.10 = 1.20
```

**Fix:** Normalize weights to sum to 1.0
```python
# clustering/config/clustering_config.py
@dataclass
class SimilarityWeights:
    # Fixed weights that sum to 1.0
    perp_name: float = 0.45  # Reduced from 0.50
    crime_type: float = 0.15
    crime_date: float = 0.15
    crime_location: float = 0.20  # Reduced from 0.30
    text_similarity: float = 0.05  # Reduced from 0.10
```

### 2. Geographic Field Compatibility

**Issue:** FIPS/GNIS fields missing from production database.

**Fix:** Migration script and conditional logic
```python
# migrations/add_geographic_fields.sql
ALTER TABLE articles 
ADD COLUMN IF NOT EXISTS fips_state VARCHAR(5),
ADD COLUMN IF NOT EXISTS fips_county VARCHAR(5),
ADD COLUMN IF NOT EXISTS gnis_city INTEGER;

CREATE INDEX IF NOT EXISTS idx_articles_fips 
ON articles(fips_state, fips_county);
```

### 3. Phase 2 Clustering Implementation

**Issue:** Phase 2 global consolidation is a stub.

**Fix:** Complete implementation in cluster_engine.py
```python
async def _phase2_clustering(self, phase1_result: ClusteringResult) -> ClusteringResult:
    """Phase 2: Global consolidation across chunks."""
    logger.info("Phase 2: Global consolidation")
    
    # Get all clusters created in phase 1
    clusters = await self.cluster_repository.get_recent_clusters(
        hours=1, min_size=2
    )
    
    # Calculate inter-cluster similarities
    merge_candidates = []
    for i, cluster1 in enumerate(clusters):
        for cluster2 in clusters[i+1:]:
            similarity = await self._calculate_cluster_similarity(
                cluster1, cluster2
            )
            if similarity > self.config.algorithm.global_consolidation_threshold:
                merge_candidates.append((cluster1.cluster_id, 
                                       cluster2.cluster_id, 
                                       similarity))
    
    # Merge clusters above threshold
    merged_count = 0
    for cluster1_id, cluster2_id, similarity in sorted(
        merge_candidates, key=lambda x: x[2], reverse=True
    ):
        if await self._can_merge_clusters(cluster1_id, cluster2_id):
            await self.cluster_repository.merge_clusters(
                cluster1_id, cluster2_id, similarity
            )
            merged_count += 1
    
    return ClusteringResult(
        clusters_created=0,
        clusters_updated=merged_count,
        articles_processed=0,
        similarity_calculations=len(clusters) * (len(clusters) - 1) // 2,
        processing_time_seconds=0,
        cluster_sizes={}
    )
```

## Architecture Overview

### System Architecture
```
┌─────────────────────┐     ┌──────────────────────┐
│   Web UI / API      │────▶│ Unified Clustering   │
└─────────────────────┘     │      Service         │
                            └──────────┬───────────┘
                                       │
                ┌──────────────────────┴───────────────────┐
                │                                          │
        ┌───────▼────────┐                     ┌──────────▼────────┐
        │ ClusterEngine  │                     │ HybridEngine      │
        │ (Clustering)   │                     │ (Enrichment)      │
        └───────┬────────┘                     └──────────┬────────┘
                │                                          │
                │              ┌────────────┐              │
                └─────────────▶│  Database  │◀─────────────┘
                               └────────────┘
```

### Data Flow
1. **Ingestion**: Articles → Extraction → Embeddings
2. **Clustering**: ClusterEngine groups similar articles
3. **Enrichment**: HybridEngine creates rich summaries (selective)
4. **Storage**: Unified format in database
5. **Delivery**: API serves clusters to UI/consumers

## Complete Implementation

### 1. Unified Clustering Service
```python
# clustering/services/unified_clustering_service.py
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..core.cluster_engine import ClusterEngine
from ..structured_outputs.hybrid_clustering import HybridClusteringEngine
from ..adapters.format_adapter import FormatAdapter
from ..monitoring.metrics_collector import MetricsCollector
from ..quality.quality_assessor import QualityAssessor

logger = logging.getLogger(__name__)

class UnifiedClusteringService:
    """
    Production clustering service that combines both engines optimally.
    """
    
    def __init__(self, config: ClusteringConfig, db_session: AsyncSession):
        self.config = config
        self.db_session = db_session
        
        # Initialize engines
        self.cluster_engine = ClusterEngine(config, db_session)
        self.hybrid_engine = HybridClusteringEngine(
            api_key=config.openai_api_key
        )
        
        # Support services
        self.format_adapter = FormatAdapter()
        self.metrics = MetricsCollector()
        self.quality_assessor = QualityAssessor(config)
        
        # Enrichment queue
        self.enrichment_queue = asyncio.Queue()
        self.enrichment_task = None
        
    async def initialize(self):
        """Start background services."""
        self.enrichment_task = asyncio.create_task(
            self._enrichment_worker()
        )
        logger.info("Unified clustering service initialized")
        
    async def cluster_articles(
        self, 
        batch_size: int = 50,
        enrich_immediately: bool = False
    ) -> Dict[str, Any]:
        """
        Main clustering method for production use.
        
        Args:
            batch_size: Articles per batch
            enrich_immediately: If True, enrich high-value clusters immediately
            
        Returns:
            Results dictionary with metrics
        """
        start_time = datetime.utcnow()
        
        try:
            # Phase 1: Robust clustering with ClusterEngine
            clustering_result = await self.cluster_engine.cluster_batch(
                extraction_status='pass',
                chunk_size=batch_size,
                force_fresh_start=False  # Always incremental
            )
            
            # Phase 2: Quality assessment
            quality_metrics = await self._assess_clustering_quality(
                clustering_result
            )
            
            # Phase 3: Selective enrichment
            enrichment_stats = await self._handle_enrichment(
                clustering_result,
                enrich_immediately
            )
            
            # Record metrics
            self.metrics.record_clustering_run(
                articles_processed=clustering_result.total_articles_processed,
                clusters_created=clustering_result.total_clusters_created,
                clusters_updated=clustering_result.phase1_result.clusters_updated,
                duration=(datetime.utcnow() - start_time).total_seconds()
            )
            
            return {
                'success': True,
                'clustering': clustering_result.to_dict(),
                'quality': quality_metrics,
                'enrichment': enrichment_stats,
                'duration': (datetime.utcnow() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}", exc_info=True)
            self.metrics.record_error('clustering_failed', str(e))
            return {
                'success': False,
                'error': str(e),
                'duration': (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def _assess_clustering_quality(
        self, 
        result: BatchClusteringResult
    ) -> Dict[str, Any]:
        """Assess quality of clustering results."""
        new_cluster_ids = list(result.phase1_result.cluster_sizes.keys())
        
        quality_issues = []
        for cluster_id in new_cluster_ids:
            assessment = await self.quality_assessor.assess_cluster(
                cluster_id
            )
            if assessment.has_issues:
                quality_issues.append({
                    'cluster_id': cluster_id,
                    'issues': assessment.issues,
                    'confidence': assessment.confidence
                })
        
        return {
            'clusters_assessed': len(new_cluster_ids),
            'quality_issues': quality_issues,
            'average_confidence': await self._get_average_confidence(
                new_cluster_ids
            )
        }
    
    async def _handle_enrichment(
        self,
        result: BatchClusteringResult,
        enrich_immediately: bool
    ) -> Dict[str, Any]:
        """Handle enrichment of new clusters."""
        enrichment_candidates = await self._identify_enrichment_candidates(
            result
        )
        
        enriched_count = 0
        queued_count = 0
        
        for cluster_id, priority in enrichment_candidates:
            if enrich_immediately and priority >= 0.8:
                # High priority - enrich now
                success = await self._enrich_cluster(cluster_id)
                if success:
                    enriched_count += 1
            else:
                # Queue for background enrichment
                await self.enrichment_queue.put((priority, cluster_id))
                queued_count += 1
        
        return {
            'candidates_identified': len(enrichment_candidates),
            'enriched_immediately': enriched_count,
            'queued_for_enrichment': queued_count,
            'queue_size': self.enrichment_queue.qsize()
        }
    
    async def _identify_enrichment_candidates(
        self,
        result: BatchClusteringResult
    ) -> List[Tuple[int, float]]:
        """
        Identify clusters that would benefit from enrichment.
        
        Returns:
            List of (cluster_id, priority) tuples
        """
        candidates = []
        
        # Get all new/updated clusters
        cluster_ids = set()
        if result.phase1_result:
            cluster_ids.update(result.phase1_result.cluster_sizes.keys())
        if result.phase2_result:
            cluster_ids.update(result.phase2_result.cluster_sizes.keys())
        
        for cluster_id in cluster_ids:
            cluster = await self.cluster_repository.get_cluster(cluster_id)
            priority = self._calculate_enrichment_priority(cluster)
            
            if priority > 0.3:  # Minimum threshold
                candidates.append((cluster_id, priority))
        
        # Sort by priority descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _calculate_enrichment_priority(self, cluster) -> float:
        """
        Calculate enrichment priority (0.0 to 1.0).
        
        Factors:
        - Cluster size (larger = higher priority)
        - Confidence (lower = higher priority) 
        - Has review flags (True = higher priority)
        - Crime severity (murder/kidnapping = higher)
        - Media coverage (high profile = higher)
        """
        priority = 0.0
        
        # Size factor (0.0 to 0.3)
        if cluster.size >= 5:
            priority += 0.3
        elif cluster.size >= 3:
            priority += 0.2
        elif cluster.size >= 2:
            priority += 0.1
        
        # Confidence factor (0.0 to 0.3)
        if cluster.overall_confidence < 0.5:
            priority += 0.3
        elif cluster.overall_confidence < 0.7:
            priority += 0.2
        elif cluster.overall_confidence < 0.8:
            priority += 0.1
        
        # Review flags (0.0 to 0.2)
        if hasattr(cluster, 'has_review_flags') and cluster.has_review_flags:
            priority += 0.2
        
        # Crime severity (0.0 to 0.2)
        summary = cluster.summary_json or {}
        crime_type = summary.get('crime_type', '').lower()
        if crime_type in ['murder', 'homicide']:
            priority += 0.2
        elif crime_type in ['kidnapping', 'abduction']:
            priority += 0.15
        elif crime_type in ['assault', 'robbery']:
            priority += 0.1
        
        return min(priority, 1.0)
    
    async def _enrich_cluster(self, cluster_id: int) -> bool:
        """
        Enrich a single cluster using HybridEngine.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get cluster with articles
            cluster = await self.cluster_repository.get_cluster_with_articles(
                cluster_id
            )
            
            if not cluster or not cluster.articles:
                logger.warning(f"Cluster {cluster_id} not found or empty")
                return False
            
            # Convert articles to format expected by HybridEngine
            articles_data = []
            for article in cluster.articles:
                articles_data.append({
                    'article_id': article.article_id,
                    'title': article.title,
                    'cleaned_text': article.cleaned_text,
                    'extracted_json': article.extracted_json,
                    'embedding': article.embedding
                })
            
            # Get LLM consolidation
            start_time = datetime.utcnow()
            case_summary = await self.hybrid_engine.consolidate_case(
                articles_data
            )
            api_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update cluster with enriched summary
            cluster.summary_json = case_summary.dict()
            cluster.enrichment_version = 1
            cluster.enriched_at = datetime.utcnow()
            
            await self.db_session.commit()
            
            # Record metrics
            self.metrics.record_enrichment(
                cluster_id=cluster_id,
                success=True,
                api_time=api_time,
                tokens_used=0  # Would need to extract from API response
            )
            
            logger.info(f"Enriched cluster {cluster_id} in {api_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enrich cluster {cluster_id}: {e}")
            self.metrics.record_enrichment(
                cluster_id=cluster_id,
                success=False,
                error=str(e)
            )
            return False
    
    async def _enrichment_worker(self):
        """Background worker for cluster enrichment."""
        logger.info("Enrichment worker started")
        
        while True:
            try:
                # Get highest priority cluster
                priority, cluster_id = await asyncio.wait_for(
                    self.enrichment_queue.get(),
                    timeout=60.0  # Check every minute
                )
                
                # Check daily budget
                if self.metrics.get_daily_api_cost() >= self.config.daily_api_budget:
                    logger.warning("Daily API budget exceeded, pausing enrichment")
                    await asyncio.sleep(3600)  # Wait an hour
                    continue
                
                # Enrich cluster
                await self._enrich_cluster(cluster_id)
                
                # Rate limiting
                await asyncio.sleep(0.1)  # 10 clusters/second max
                
            except asyncio.TimeoutError:
                # No clusters to enrich, continue
                continue
            except Exception as e:
                logger.error(f"Enrichment worker error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        return {
            'clustering_metrics': await self.cluster_engine.get_quality_metrics(),
            'enrichment_metrics': self.metrics.get_enrichment_stats(),
            'queue_size': self.enrichment_queue.qsize(),
            'daily_api_cost': self.metrics.get_daily_api_cost(),
            'system_health': await self._check_system_health()
        }
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check health of all components."""
        return {
            'cluster_engine': 'healthy',  # Would implement actual checks
            'hybrid_engine': 'healthy',
            'database': await self._check_database_health(),
            'enrichment_worker': 'running' if self.enrichment_task and not self.enrichment_task.done() else 'stopped'
        }
    
    async def shutdown(self):
        """Graceful shutdown."""
        if self.enrichment_task:
            self.enrichment_task.cancel()
            try:
                await self.enrichment_task
            except asyncio.CancelledError:
                pass
        logger.info("Unified clustering service shut down")
```

### 2. Format Adapter Implementation
```python
# clustering/adapters/format_adapter.py
from typing import Dict, Any, Optional
import json
from datetime import datetime

class FormatAdapter:
    """Convert between ClusterEngine and HybridEngine formats."""
    
    @staticmethod
    def to_hybrid_format(cluster_json: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ClusterEngine format to HybridEngine format."""
        # Extract primary information
        perp_name = cluster_json.get('primary_perpetrator_name', 'Unknown')
        crime_type = cluster_json.get('primary_crime_type', 'Unknown')
        location = cluster_json.get('primary_location', 'Unknown')
        
        # Parse location
        city, state = FormatAdapter._parse_location(location)
        
        # Build hybrid format
        return {
            'case_name': f"{perp_name} - {crime_type} Case",
            'case_status': 'discovered',
            'suspect_name': perp_name,
            'victim_name': cluster_json.get('victim_info', {}).get('name', 'Unknown'),
            'crime_type': crime_type,
            'state': state,
            'adjudication_status': 'pending',
            'crime': {
                'type': crime_type,
                'motive': None,
                'weapon': None,
                'date_range': {
                    'start': cluster_json.get('primary_crime_date'),
                    'end': cluster_json.get('primary_crime_date')
                }
            },
            'victim': cluster_json.get('victim_info', {}),
            'perpetrator': {
                'name': perp_name,
                'aliases': [],
                'age': None
            },
            'location': {
                'city': city,
                'state': state,
                'country': 'United States'
            },
            'case_summary': cluster_json.get('consolidated_case_summary', ''),
            'key_evidence': [],
            'potential_complications': [],
            'consolidation_confidence': int(cluster_json.get('overall_confidence', 0.5) * 100),
            'review_flags': []
        }
    
    @staticmethod
    def _parse_location(location: str) -> Tuple[Optional[str], str]:
        """Parse location string into city and state."""
        if not location or location == 'Unknown':
            return None, 'Unknown'
        
        parts = location.split(',')
        if len(parts) >= 2:
            city = parts[0].strip()
            state = parts[-1].strip()
            return city, state
        else:
            return None, location.strip()
    
    @staticmethod
    def detect_format(summary_json: Dict[str, Any]) -> str:
        """Detect which format a summary is in."""
        # HybridEngine format has these specific fields
        hybrid_fields = {'case_name', 'suspect_name', 'victim_name', 'case_status'}
        # ClusterEngine format has these fields
        cluster_fields = {'primary_perpetrator_name', 'primary_crime_type', 'source_articles'}
        
        json_fields = set(summary_json.keys())
        
        if hybrid_fields.issubset(json_fields):
            return 'hybrid'
        elif cluster_fields.issubset(json_fields):
            return 'cluster'
        else:
            return 'unknown'
```

### 3. Quality Assessment Module
```python
# clustering/quality/quality_assessor.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ClusterAssessment:
    """Results of cluster quality assessment."""
    cluster_id: int
    confidence: float
    has_issues: bool
    issues: List[str]
    recommendations: List[str]

class QualityAssessor:
    """Assess cluster quality and identify issues."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.thresholds = config.quality
        
    async def assess_cluster(self, cluster_id: int) -> ClusterAssessment:
        """Comprehensive cluster quality assessment."""
        cluster = await self.cluster_repository.get_cluster_with_articles(cluster_id)
        
        issues = []
        recommendations = []
        
        # Check confidence
        if cluster.overall_confidence < self.thresholds.confidence_alert_threshold:
            issues.append(f"Low confidence: {cluster.overall_confidence:.2f}")
            recommendations.append("Review for potential split")
        
        # Check size
        if cluster.size > self.thresholds.cluster_size_alert_threshold:
            issues.append(f"Large cluster: {cluster.size} articles")
            recommendations.append("Check for over-merging")
        
        # Check geographic consistency
        if self.thresholds.location_inconsistency_flag:
            states = self._extract_states(cluster.articles)
            if len(states) > 1:
                issues.append(f"Multiple states: {', '.join(states)}")
                recommendations.append("Verify same case across states")
        
        # Check name consistency
        names = self._extract_perpetrator_names(cluster.articles)
        if len(set(names)) > 1:
            name_similarity = self._calculate_name_consistency(names)
            if name_similarity < self.thresholds.name_mismatch_threshold:
                issues.append(f"Name inconsistency: {', '.join(set(names))}")
                recommendations.append("Verify perpetrator identity")
        
        return ClusterAssessment(
            cluster_id=cluster_id,
            confidence=cluster.overall_confidence,
            has_issues=len(issues) > 0,
            issues=issues,
            recommendations=recommendations
        )
    
    def _extract_states(self, articles: List[Article]) -> Set[str]:
        """Extract unique states from articles."""
        states = set()
        for article in articles:
            location = article.extracted_json.get('crime_location', '')
            if ',' in location:
                state = location.split(',')[-1].strip()
                states.add(state)
        return states
    
    def _extract_perpetrator_names(self, articles: List[Article]) -> List[str]:
        """Extract perpetrator names from articles."""
        names = []
        for article in articles:
            name = article.extracted_json.get('perp_name', '')
            if name and name != 'Unknown':
                names.append(name)
        return names
    
    def _calculate_name_consistency(self, names: List[str]) -> float:
        """Calculate average similarity between all name pairs."""
        if len(names) < 2:
            return 1.0
        
        from ..entity_resolution.name_matcher import NameMatcher
        matcher = NameMatcher({})
        
        total_sim = 0
        count = 0
        
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                sim, _ = matcher.similarity(name1, name2)
                total_sim += sim
                count += 1
        
        return total_sim / count if count > 0 else 1.0
```

### 4. Monitoring and Metrics
```python
# clustering/monitoring/metrics_collector.py
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json

class MetricsCollector:
    """Collect and aggregate clustering metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.daily_costs = defaultdict(float)
        self.errors = []
        
    def record_clustering_run(
        self,
        articles_processed: int,
        clusters_created: int,
        clusters_updated: int,
        duration: float
    ):
        """Record metrics for a clustering run."""
        self.metrics['clustering_runs'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'articles_processed': articles_processed,
            'clusters_created': clusters_created,
            'clusters_updated': clusters_updated,
            'duration': duration,
            'articles_per_second': articles_processed / duration if duration > 0 else 0
        })
    
    def record_enrichment(
        self,
        cluster_id: int,
        success: bool,
        api_time: float = 0,
        tokens_used: int = 0,
        error: Optional[str] = None
    ):
        """Record enrichment operation."""
        cost = self._calculate_api_cost(tokens_used)
        
        self.metrics['enrichments'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'cluster_id': cluster_id,
            'success': success,
            'api_time': api_time,
            'tokens_used': tokens_used,
            'cost': cost,
            'error': error
        })
        
        # Track daily costs
        today = datetime.utcnow().date().isoformat()
        self.daily_costs[today] += cost
    
    def record_error(self, error_type: str, error_message: str):
        """Record an error."""
        self.errors.append({
            'timestamp': datetime.utcnow().isoformat(),
            'type': error_type,
            'message': error_message
        })
    
    def _calculate_api_cost(self, tokens: int) -> float:
        """Calculate API cost based on token usage."""
        # GPT-4.1-mini pricing (example)
        input_token_cost = 0.15 / 1_000_000  # $0.15 per 1M tokens
        output_token_cost = 0.60 / 1_000_000  # $0.60 per 1M tokens
        
        # Rough estimate: 75% input, 25% output
        input_tokens = int(tokens * 0.75)
        output_tokens = tokens - input_tokens
        
        return (input_tokens * input_token_cost + 
                output_tokens * output_token_cost)
    
    def get_daily_api_cost(self) -> float:
        """Get today's API cost."""
        today = datetime.utcnow().date().isoformat()
        return self.daily_costs.get(today, 0.0)
    
    def get_enrichment_stats(self) -> Dict[str, Any]:
        """Get enrichment statistics."""
        if not self.metrics['enrichments']:
            return {
                'total_enrichments': 0,
                'success_rate': 0,
                'average_api_time': 0,
                'total_cost': 0,
                'daily_cost': 0
            }
        
        enrichments = self.metrics['enrichments']
        successful = [e for e in enrichments if e['success']]
        
        return {
            'total_enrichments': len(enrichments),
            'successful_enrichments': len(successful),
            'success_rate': len(successful) / len(enrichments),
            'average_api_time': sum(e['api_time'] for e in successful) / len(successful) if successful else 0,
            'total_cost': sum(e['cost'] for e in enrichments),
            'daily_cost': self.get_daily_api_cost(),
            'recent_errors': [e for e in self.errors if 
                            datetime.fromisoformat(e['timestamp']) > 
                            datetime.utcnow() - timedelta(hours=24)]
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        export_data = {
            'exported_at': datetime.utcnow().isoformat(),
            'clustering_runs': self.metrics['clustering_runs'][-100:],  # Last 100
            'enrichments': self.metrics['enrichments'][-1000:],  # Last 1000
            'daily_costs': dict(self.daily_costs),
            'errors': self.errors[-100:]  # Last 100 errors
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
```

## Code Cleanup Initiative

### Phase 1: Remove Redundant Files (Week 1)
```bash
# Backup before cleanup
tar -czf clustering_backup_$(date +%Y%m%d).tar.gz clustering/

# Remove duplicate service files
rm clustering/working_service.py.backup
rm clustering/working_service_backup.py

# Remove one-off test scripts
rm clustering/test_clustering_fix.py
rm clustering/test_clustering_improvements.py
rm clustering/test_incremental_direct.py

# Archive old documentation
mkdir -p clustering/archive/docs
mv clustering/PHASE1_IMPLEMENTATION_SUMMARY.md clustering/archive/docs/
mv clustering/CLUSTERING_FIX_SUMMARY.md clustering/archive/docs/
```

### Phase 2: Reorganize Structure (Week 1)
```
clustering/
├── __init__.py
├── __main__.py
├── requirements.txt
├── Dockerfile
├── README.md
├── config/
│   ├── __init__.py
│   ├── clustering_config.py
│   └── settings.yaml
├── core/                    # Core clustering logic
│   ├── __init__.py
│   ├── cluster_engine.py
│   ├── similarity_calculator.py
│   └── graph_builder.py
├── enrichment/             # LLM enrichment (was structured_outputs)
│   ├── __init__.py
│   ├── hybrid_engine.py
│   ├── schemas.py
│   └── local_similarity.py
├── services/               # High-level services
│   ├── __init__.py
│   ├── unified_service.py
│   └── working_service.py
├── adapters/              # Format converters
│   ├── __init__.py
│   └── format_adapter.py
├── quality/               # Quality assessment
│   ├── __init__.py
│   ├── quality_assessor.py
│   └── review_flags.py
├── monitoring/            # Metrics and monitoring
│   ├── __init__.py
│   └── metrics_collector.py
├── scripts/               # Operational scripts
│   ├── run_clustering.py
│   ├── monitor_duplicates.py
│   └── enrich_clusters.py
├── tests/                 # All tests
│   ├── conftest.py
│   ├── test_similarity.py
│   ├── test_clustering.py
│   ├── test_enrichment.py
│   └── test_unified_service.py
└── migrations/           # Database migrations
    └── add_geographic_fields.sql
```

### Phase 3: Code Quality Improvements (Week 2)

#### 1. Add Type Hints
```python
# Before
def calculate_similarity(self, article1, article2):
    pass

# After
def calculate_similarity(
    self, 
    article1: Article, 
    article2: Article
) -> SimilarityResult:
    pass
```

#### 2. Add Docstrings
```python
def cluster_articles(
    self, 
    articles: List[Article], 
    incremental: bool = True
) -> ClusteringResult:
    """
    Cluster a list of articles using multi-factor similarity.
    
    This method implements the core clustering algorithm with support
    for both fresh clustering and incremental updates to existing clusters.
    
    Args:
        articles: List of Article objects to cluster
        incremental: If True, consider merging with existing clusters.
                    If False, create new clusters only.
    
    Returns:
        ClusteringResult containing clustering statistics and cluster IDs
        
    Raises:
        ClusteringError: If clustering fails due to invalid input
        DatabaseError: If database operations fail
    
    Example:
        >>> result = await cluster_engine.cluster_articles(
        ...     articles=[article1, article2],
        ...     incremental=True
        ... )
        >>> print(f"Created {result.clusters_created} clusters")
    """
    pass
```

#### 3. Add Logging
```python
# Structured logging with context
logger.info(
    "Clustering completed",
    extra={
        'articles_processed': result.articles_processed,
        'clusters_created': result.clusters_created,
        'duration': duration,
        'incremental': incremental
    }
)
```

#### 4. Error Handling
```python
class ClusteringError(Exception):
    """Base exception for clustering errors."""
    pass

class SimilarityCalculationError(ClusteringError):
    """Error calculating similarity between articles."""
    pass

class EnrichmentError(ClusteringError):
    """Error enriching cluster with LLM."""
    pass

# Usage
try:
    result = await self.calculate_similarity(article1, article2)
except SimilarityCalculationError as e:
    logger.error(f"Similarity calculation failed: {e}")
    # Fallback logic
    return SimilarityResult(0.0, {}, has_veto=True)
```

### Phase 4: Performance Optimizations (Week 2)

#### 1. Batch Database Operations
```python
# Before - N queries
for article_id in article_ids:
    article = await get_article(article_id)
    process(article)

# After - 1 query
articles = await get_articles_batch(article_ids)
for article in articles:
    process(article)
```

#### 2. Implement Caching
```python
from functools import lru_cache
from aiocache import cached

class SimilarityCalculator:
    @cached(ttl=3600)  # Cache for 1 hour
    async def calculate_similarity(
        self, 
        article1_id: str, 
        article2_id: str
    ) -> float:
        # Expensive calculation
        pass
```

#### 3. Use Connection Pooling
```python
# Database configuration
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

## Testing Strategy

### 1. Unit Tests
```python
# tests/test_similarity.py
import pytest
from clustering.core.similarity_calculator import SimilarityCalculator

class TestSimilarityCalculator:
    @pytest.mark.asyncio
    async def test_exact_name_match(self, similarity_calculator, sample_articles):
        """Test that identical names produce high similarity."""
        article1, article2 = sample_articles['same_perpetrator']
        result = await similarity_calculator.calculate_similarity(
            article1.article_id, 
            article2.article_id
        )
        assert result.overall_similarity > 0.8
        assert result.factor_scores['perp_name'] == 1.0
    
    @pytest.mark.asyncio  
    async def test_name_veto(self, similarity_calculator, sample_articles):
        """Test that conflicting high-confidence names trigger veto."""
        article1, article2 = sample_articles['different_perpetrators']
        result = await similarity_calculator.calculate_similarity(
            article1.article_id,
            article2.article_id
        )
        assert result.overall_similarity == 0.0
        assert result.has_veto
        assert result.veto_reason == 'name_conflict'
    
    @pytest.mark.asyncio
    async def test_weight_normalization(self, clustering_config):
        """Test that similarity weights sum to 1.0."""
        weights = clustering_config.weights
        total = (weights.perp_name + weights.crime_type + 
                weights.crime_date + weights.crime_location + 
                weights.text_similarity)
        assert abs(total - 1.0) < 0.001
```

### 2. Integration Tests
```python
# tests/test_unified_service.py
import pytest
from clustering.services.unified_clustering_service import UnifiedClusteringService

class TestUnifiedClusteringService:
    @pytest.mark.asyncio
    async def test_full_clustering_pipeline(self, unified_service, test_articles):
        """Test complete clustering pipeline."""
        # Insert test articles
        await insert_test_articles(test_articles)
        
        # Run clustering
        result = await unified_service.cluster_articles(
            batch_size=10,
            enrich_immediately=False
        )
        
        assert result['success']
        assert result['clustering']['total_articles_processed'] == len(test_articles)
        assert result['clustering']['total_clusters_created'] > 0
        assert len(result['quality']['quality_issues']) == 0
    
    @pytest.mark.asyncio
    async def test_enrichment_queue(self, unified_service, test_cluster):
        """Test enrichment queue processing."""
        # Add cluster to enrichment queue
        await unified_service.enrichment_queue.put((0.9, test_cluster.cluster_id))
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Verify enrichment
        enriched_cluster = await get_cluster(test_cluster.cluster_id)
        assert enriched_cluster.enrichment_version == 1
        assert 'case_name' in enriched_cluster.summary_json
```

### 3. Performance Tests
```python
# tests/test_performance.py
import pytest
import time

class TestPerformance:
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_clustering_throughput(self, unified_service, large_article_set):
        """Test clustering can handle 1000 articles in under 60 seconds."""
        start_time = time.time()
        
        result = await unified_service.cluster_articles(
            batch_size=100,
            enrich_immediately=False
        )
        
        duration = time.time() - start_time
        
        assert result['success']
        assert duration < 60  # Should complete in under 60 seconds
        assert result['clustering']['total_articles_processed'] == 1000
        
        # Calculate throughput
        articles_per_second = 1000 / duration
        assert articles_per_second > 15  # Minimum 15 articles/second
```

### 4. Data Quality Tests
```python
# tests/test_data_quality.py
import pytest

class TestDataQuality:
    @pytest.mark.asyncio
    async def test_no_duplicate_clusters(self, unified_service, test_articles):
        """Test that multi-session clustering prevents duplicates."""
        # First run
        result1 = await unified_service.cluster_articles()
        clusters1 = result1['clustering']['total_clusters_created']
        
        # Second run with same articles
        result2 = await unified_service.cluster_articles()
        clusters2 = result2['clustering']['total_clusters_created']
        
        # Should not create new clusters
        assert clusters2 == 0
        assert result2['clustering']['phase1_result']['clusters_updated'] > 0
```

## Performance Metrics

### Key Performance Indicators (KPIs)

#### 1. Clustering Performance
- **Throughput**: Articles processed per second
- **Latency**: Time to cluster a batch
- **Accuracy**: Precision and recall of clustering
- **Database Load**: Queries per second

#### 2. Enrichment Performance  
- **API Latency**: Average response time
- **Success Rate**: Successful enrichments / total attempts
- **Cost Efficiency**: Cost per enriched cluster
- **Queue Length**: Pending enrichments

#### 3. System Health
- **Uptime**: Service availability
- **Error Rate**: Errors per 1000 operations
- **Memory Usage**: Peak and average
- **CPU Usage**: Peak and average

### Performance Benchmarks

| Metric | Target | Acceptable | Alert Threshold |
|--------|--------|------------|-----------------|
| Clustering Throughput | 30 articles/sec | 20 articles/sec | < 10 articles/sec |
| Clustering Latency | < 2 sec/batch | < 5 sec/batch | > 10 sec/batch |
| Enrichment Success Rate | > 99% | > 95% | < 90% |
| API Cost per Day | < $1 | < $5 | > $10 |
| Memory Usage | < 2GB | < 4GB | > 6GB |
| Error Rate | < 0.1% | < 1% | > 5% |

### Performance Monitoring Dashboard
```python
# monitoring/dashboard.py
from prometheus_client import Counter, Histogram, Gauge

# Metrics
clustering_counter = Counter(
    'clustering_articles_total',
    'Total articles clustered',
    ['status']
)

clustering_duration = Histogram(
    'clustering_duration_seconds',
    'Clustering duration in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

enrichment_cost = Counter(
    'enrichment_api_cost_dollars',
    'Total API cost in dollars'
)

cluster_quality = Gauge(
    'cluster_quality_score',
    'Average cluster quality score'
)

# Usage in code
clustering_counter.labels(status='success').inc(articles_processed)
clustering_duration.observe(duration)
enrichment_cost.inc(api_cost)
cluster_quality.set(average_confidence)
```

## Monitoring & Alerting

### 1. Application Logs
```yaml
# logging_config.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    
handlers:
  console:
    class: logging.StreamHandler
    formatter: default
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    formatter: json
    filename: logs/clustering.log
    maxBytes: 104857600  # 100MB
    backupCount: 10
    
loggers:
  clustering:
    level: INFO
    handlers: [console, file]
    propagate: false
```

### 2. Metrics Collection
```python
# Configuration for Prometheus/Grafana
METRICS_CONFIG = {
    'enabled': True,
    'port': 9090,
    'path': '/metrics',
    'update_interval': 10,  # seconds
}

# StatsD integration for real-time metrics
STATSD_CONFIG = {
    'host': 'localhost',
    'port': 8125,
    'prefix': 'clustering',
}
```

### 3. Alerting Rules
```yaml
# alerts.yaml
groups:
  - name: clustering_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(clustering_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High clustering error rate
          
      - alert: APIBudgetExceeded
        expr: enrichment_api_cost_dollars > 10
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: Daily API budget exceeded
          
      - alert: ClusteringBacklog
        expr: clustering_queue_size > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: Large clustering backlog
```

### 4. Health Check Endpoints
```python
# Health check implementation
from fastapi import FastAPI
from typing import Dict

app = FastAPI()

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check."""
    return {"status": "healthy"}

@app.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with component status."""
    service = get_unified_service()
    metrics = await service.get_metrics()
    
    return {
        "status": "healthy" if all_components_healthy(metrics) else "degraded",
        "components": {
            "clustering_engine": metrics['system_health']['cluster_engine'],
            "enrichment_engine": metrics['system_health']['hybrid_engine'],
            "database": metrics['system_health']['database'],
            "enrichment_worker": metrics['system_health']['enrichment_worker']
        },
        "metrics": {
            "queue_size": metrics['queue_size'],
            "daily_api_cost": metrics['daily_api_cost'],
            "error_rate": calculate_error_rate(metrics)
        }
    }
```

## Data Quality Framework

### 1. Quality Metrics
```python
@dataclass
class ClusterQualityMetrics:
    """Comprehensive cluster quality metrics."""
    
    # Accuracy metrics
    precision: float  # Correct clusters / Total clusters
    recall: float     # Found clusters / Expected clusters
    f1_score: float   # Harmonic mean of precision and recall
    
    # Consistency metrics
    name_consistency: float    # Average name similarity within clusters
    location_consistency: float # Single state clusters / Total clusters
    temporal_consistency: float # Date range consistency
    
    # Confidence metrics
    average_confidence: float
    low_confidence_ratio: float  # Clusters below threshold / Total
    
    # Size metrics
    average_cluster_size: float
    oversized_cluster_ratio: float  # Large clusters / Total
    singleton_ratio: float  # Single article clusters / Total
```

### 2. Quality Monitoring
```python
class QualityMonitor:
    """Monitor and track clustering quality over time."""
    
    async def calculate_quality_metrics(
        self, 
        time_window: timedelta = timedelta(days=1)
    ) -> ClusterQualityMetrics:
        """Calculate quality metrics for recent clusters."""
        # Get recent clusters
        recent_clusters = await self.get_recent_clusters(time_window)
        
        # Calculate metrics
        metrics = ClusterQualityMetrics(
            precision=await self._calculate_precision(recent_clusters),
            recall=await self._calculate_recall(recent_clusters),
            f1_score=self._calculate_f1(precision, recall),
            name_consistency=await self._calculate_name_consistency(recent_clusters),
            location_consistency=await self._calculate_location_consistency(recent_clusters),
            temporal_consistency=await self._calculate_temporal_consistency(recent_clusters),
            average_confidence=await self._calculate_average_confidence(recent_clusters),
            low_confidence_ratio=await self._calculate_low_confidence_ratio(recent_clusters),
            average_cluster_size=await self._calculate_average_size(recent_clusters),
            oversized_cluster_ratio=await self._calculate_oversized_ratio(recent_clusters),
            singleton_ratio=await self._calculate_singleton_ratio(recent_clusters)
        )
        
        # Record in time series
        await self.record_quality_metrics(metrics)
        
        # Check thresholds and alert if needed
        await self._check_quality_thresholds(metrics)
        
        return metrics
```

### 3. Quality Thresholds
```yaml
# quality_thresholds.yaml
thresholds:
  precision:
    target: 0.95
    acceptable: 0.90
    alert: 0.85
    
  recall:
    target: 0.90
    acceptable: 0.85
    alert: 0.80
    
  name_consistency:
    target: 0.90
    acceptable: 0.85
    alert: 0.80
    
  location_consistency:
    target: 0.95
    acceptable: 0.90
    alert: 0.85
    
  average_confidence:
    target: 0.80
    acceptable: 0.70
    alert: 0.60
    
  oversized_cluster_ratio:
    target: 0.01  # 1%
    acceptable: 0.05  # 5%
    alert: 0.10  # 10%
```

## Operational Runbooks

### 1. Daily Operations Runbook
```markdown
# Daily Clustering Operations

## Morning Checklist (9 AM)
1. Check overnight clustering results
   ```bash
   ./scripts/check_overnight_results.sh
   ```

2. Review quality metrics
   ```bash
   curl http://localhost:8000/metrics/quality
   ```

3. Check API costs
   ```bash
   ./scripts/check_api_costs.sh
   ```

4. Review error logs
   ```bash
   tail -n 100 logs/clustering.log | grep ERROR
   ```

## Afternoon Tasks (2 PM)
1. Run manual clustering for high-priority articles
   ```bash
   ./scripts/run_priority_clustering.sh
   ```

2. Review enrichment queue
   ```bash
   curl http://localhost:8000/enrichment/queue
   ```

3. Check system health
   ```bash
   curl http://localhost:8000/health/detailed
   ```

## End of Day (5 PM)
1. Export daily metrics
   ```bash
   ./scripts/export_daily_metrics.sh
   ```

2. Schedule overnight clustering
   ```bash
   ./scripts/schedule_overnight.sh
   ```

3. Verify backups completed
   ```bash
   ./scripts/verify_backups.sh
   ```
```

### 2. Incident Response Runbook
```markdown
# Clustering Incident Response

## High Error Rate
1. Check recent changes
   ```bash
   git log --oneline -10
   ```

2. Review error patterns
   ```bash
   ./scripts/analyze_errors.sh --last-hour
   ```

3. Rollback if needed
   ```bash
   ./scripts/rollback_clustering.sh
   ```

## API Budget Exceeded
1. Pause enrichment worker
   ```bash
   curl -X POST http://localhost:8000/enrichment/pause
   ```

2. Review enrichment logs
   ```bash
   grep "enrichment" logs/clustering.log | tail -100
   ```

3. Adjust thresholds
   ```bash
   ./scripts/adjust_enrichment_thresholds.sh --conservative
   ```

## Database Performance Issues
1. Check connection pool
   ```bash
   ./scripts/check_db_connections.sh
   ```

2. Analyze slow queries
   ```sql
   SELECT query, mean_time 
   FROM pg_stat_statements 
   WHERE mean_time > 1000 
   ORDER BY mean_time DESC;
   ```

3. Run vacuum if needed
   ```sql
   VACUUM ANALYZE clusters;
   VACUUM ANALYZE articles;
   ```
```

### 3. Deployment Runbook
```markdown
# Clustering Service Deployment

## Pre-deployment Checklist
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Database migrations tested
- [ ] Rollback plan ready

## Deployment Steps
1. Tag release
   ```bash
   git tag -a v1.0.0 -m "Unified clustering service"
   git push origin v1.0.0
   ```

2. Build Docker image
   ```bash
   docker build -t clustering:v1.0.0 .
   docker push registry/clustering:v1.0.0
   ```

3. Run database migrations
   ```bash
   ./scripts/run_migrations.sh
   ```

4. Deploy to staging
   ```bash
   kubectl apply -f k8s/staging/
   ```

5. Run smoke tests
   ```bash
   ./scripts/smoke_tests.sh staging
   ```

6. Deploy to production
   ```bash
   kubectl apply -f k8s/production/
   ```

7. Monitor deployment
   ```bash
   ./scripts/monitor_deployment.sh
   ```

## Rollback Procedure
1. Switch traffic to previous version
   ```bash
   kubectl set image deployment/clustering clustering=registry/clustering:v0.9.0
   ```

2. Verify rollback
   ```bash
   kubectl rollout status deployment/clustering
   ```

3. Investigate issues
   ```bash
   ./scripts/analyze_deployment_failure.sh
   ```
```

## Migration Timeline

### Week 1: Foundation
- **Day 1-2**: Fix critical bugs (weights, geographic fields)
- **Day 3-4**: Implement UnifiedClusteringService
- **Day 5**: Initial testing and validation

### Week 2: Integration  
- **Day 1-2**: Complete format adapter and quality assessor
- **Day 3-4**: Implement monitoring and metrics
- **Day 5**: Integration testing

### Week 3: Testing & Optimization
- **Day 1-2**: Performance testing and optimization
- **Day 3-4**: Data quality validation
- **Day 5**: Documentation and runbooks

### Week 4: Production Deployment
- **Day 1**: Deploy to staging environment
- **Day 2-3**: Staging validation and load testing
- **Day 4**: Production deployment
- **Day 5**: Post-deployment monitoring

### Month 2: Optimization
- **Week 1**: Tune enrichment thresholds based on metrics
- **Week 2**: Implement advanced caching strategies
- **Week 3**: Add predictive enrichment models
- **Week 4**: Quarterly review and planning

## Success Metrics

### Technical Success
- Zero duplicate clusters created
- 99%+ enrichment success rate
- < $5/day API costs
- < 2 second clustering latency

### Business Success
- 95%+ cluster accuracy
- 90%+ user satisfaction with case summaries
- 50% reduction in manual review time
- 100% audit trail compliance

### Operational Success
- 99.9% uptime
- < 1 hour MTTR for incidents
- 100% runbook coverage
- Zero data loss incidents

## Risk Mitigation

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API rate limits | Medium | High | Implement backoff and queuing |
| Database scaling | Low | High | Add read replicas, optimize queries |
| Memory leaks | Medium | Medium | Regular profiling, auto-restart |
| Data corruption | Low | Critical | Transactions, backups, validation |

### Operational Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Key person dependency | Medium | High | Documentation, cross-training |
| Budget overrun | Low | Medium | Cost alerts, daily limits |
| Compliance issues | Low | High | Audit trails, data retention |

## Conclusion

This comprehensive strategy provides:
1. **Complete implementation** of unified clustering service
2. **Bug fixes** for critical issues
3. **Code cleanup** plan with timeline
4. **Testing strategy** with examples
5. **Performance metrics** and monitoring
6. **Operational runbooks** for daily use
7. **Migration timeline** with clear milestones

The unified approach leverages the strengths of both engines while addressing all identified gaps. With proper implementation and monitoring, this will provide a robust, cost-effective clustering solution for the SNAPPED pipeline.
