# 📊 Code Quality Report

Generated: 2025-11-28 22:49
Source: `src/workflow_composer`

---

## 📋 Summary

| Category | Count | Priority |
|----------|-------|----------|
| 💀 Dead Code (>80% confidence) | 25 | Medium |
| 🔄 Duplicate Code Groups | 15 | High |
| 📊 High Complexity Functions | 104 | Medium |

### Recommended Actions

1. **Duplicate Code**: Review each duplicate group. Consolidate if truly identical.
2. **Dead Code**: Verify with `git log` - if unused for >6 months, consider removal.
3. **Complexity**: Break down C/D/E/F functions into smaller units.

## 🔄 Duplicate Code Detection (Pylint)

Pylint identifies similar code blocks (min 10 lines).
Review each case - some duplication may be intentional.

```
src/workflow_composer/results/collector.py:1:0: R0801: Similar lines in 2 files
==workflow_composer.agents.tools.data_discovery:[55:152]
==workflow_composer.agents.tools:[480:593]
        if scanner is None:
            return ToolResult(
                success=False,
                tool_name="scan_data",
                error="Scanner not available",
                message="❌ Data scanner is not available. Please check installation."
            )
        # Smart default paths - check common data locations
        # Development defaults for sdodl001's environment
        if not path:
            default_paths = [
                Path("/scratch/sdodl001/BioPipelines"),  # Primary data location
                Path("/scratch/sdodl001/BioPipelines/data"),
                Path.home() / "BioPipelines" / "data",
                Path.home() / "data",
                Path.cwd() / "data",
                Path.cwd(),
            ]
            for p in default_paths:
                if p.exists() and p.is_dir():
                    path = str(p)
                    break
            else:
                path = str(Path.cwd())
        # Clean up path
        path = path.strip().strip("'\"")
        scan_path = Path(path).expanduser().resolve()
        if not scan_path.exists():
            return ToolResult(
                success=False,
                tool_name="scan_data",
                error=f"Path not found: {scan_path}",
                message=f"❌ Directory not found: `{scan_path}`"
            )
        try:
            result = scanner.scan_directory(scan_path, recursive=True)
            samples = result.samples if hasattr(result, 'samples') else []
            # Add samples to manifest
            if manifest and samples:
                for sample in samples:
                    manifest.add_sample(sample)
            # Build response message
            if samples:
                sample_list = []
                for s in samples[:10]:
                    # Count files (1 for single-end, 2 for paired-end)
                    file_count = 2 if (hasattr(s, 'is_paired') and s.is_paired) or (hasattr(s, 'fastq_2') and s.fastq_2) else 1
                    # Get layout string
                    layout = "paired" if file_count == 2 else "single"
                    if hasattr(s, 'library_layout'):
                        layout = s.library_layout.value if hasattr(s.library_layout, 'value') else str(s.library_layout)
                    sample_list.append(f"  - `{s.sample_id}`: {file_count} files ({layout})")
                sample_str = "\n".join(sample_list)
                if len(samples) > 10:
                    sample_str += f"\n  - ... and {len(samples) - 10} more"
                message = f"""✅ Found **{len(samples)} samples** in `{scan_path}`:
{sample_str}
Added to data manifest. Ready for workflow generation!"""
            else:
                message = f"⚠️ No FASTQ samples found in `{scan_path}`"
            return ToolResult(
                success=True,
                tool_name="scan_data",
                data={
                    "samples": samples,
                    "path": str(scan_path),
                    "count": len(samples)
                },
                message=message,
                ui_update={
                    "manifest_sample_count": len(samples),
                    "manifest_path": str(scan_path)
                }
            )
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return ToolResult(
                success=False,
                tool_name="scan_data",
                error=str(e),
                message=f"❌ Failed to scan directory: {e}"
            )
    def cleanup_data(self, path: str = None, confirm: bool = False) -> ToolResult:
        """
        Clean up corrupted data files (HTML error pages masquerading as FASTQ, etc).
        Two-phase operation:
        1. First call (confirm=False): Scan and show what would be deleted
        2. Second call (confirm=True): Actually delete the files
        Args:
            path: Directory path to clean. Defaults to data directory.
            confirm: If True, actually delete files. If False, just show preview.
        Returns:
            ToolResult with cleanup summary or preview
        """
        import gzip
        # Use default data path if not specified (duplicate-code)
---
src/workflow_composer/results/collector.py:1:0: R0801: Similar lines in 2 files
==workflow_composer.agents.tools.data_discovery:[190:241]
==workflow_composer.agents.tools:[885:938]
            search_query = SearchQueryModel(raw_query=query, max_results=5)
        # Search ENCODE
        try:
            encode = ENCODEAdapter()
            encode_query = SearchQueryModel(
                raw_query=search_query.raw_query or query,
                organism=search_query.organism,
                assay_type=search_query.assay_type,
                target=search_query.target,
                tissue=search_query.tissue,
                cell_line=search_query.cell_line,
                max_results=5
            )
            encode_results = encode.search(encode_query)
            if encode_results:
                for dataset in encode_results[:5]:
                    results.append({
                        "source": "ENCODE",
                        "id": dataset.id,
                        "title": dataset.title or dataset.id,
                        "organism": dataset.organism or "",
                        "assay": dataset.assay_type or ""
                    })
        except Exception as e:
            logger.debug(f"ENCODE search failed: {e}")
        # Search GEO
        try:
            geo = GEOAdapter()
            geo_query = SearchQueryModel(
                raw_query=search_query.raw_query or query,
                organism=search_query.organism,
                assay_type=search_query.assay_type,
                tissue=search_query.tissue,
                max_results=5
            )
            geo_results = geo.search(geo_query)
            if geo_results:
                for dataset in geo_results[:5]:
                    results.append({
                        "source": "GEO",
                        "id": dataset.id,
                        "title": dataset.title or dataset.id,
                        "organism": dataset.organism or "",
                        "assay": dataset.assay_type or ""
                    })
        except Exception as e:
            logger.debug(f"GEO search failed: {e}")
        if results: (duplicate-code)
---
src/workflow_composer/results/collector.py:1:0: R0801: Similar lines in 2 files
==workflow_composer.models.providers.lightning:[63:116]
==workflow_composer.models.providers.openai:[63:115]
                data = await response.json()
        # Extract response
        content = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return {
            "content": content,
            "tokens_used": tokens_used,
            "model": model,
            "raw_response": data,
        }
    async def health_check(self) -> Dict[str, Any]:
        """Check if OpenAI is available."""
        if not self.api_key:
            return {
                "available": False,
                "error": "API key not configured",
            }
        try:
            start = time.time()
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    latency = (time.time() - start) * 1000
                    if response.status == 200:
                        return {
                            "available": True,
                            "latency_ms": latency,
                        }
                    else:
                        return {
                            "available": False,
                            "error": f"HTTP {response.status}",
                            "latency_ms": latency,
                        }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
            } (duplicate-code)
---
src/workflow_composer/results/collector.py:1:0: R0801: Similar lines in 2 files
==workflow_composer.providers.lightning:[183:222]
==workflow_composer.providers.openai:[111:150]
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if response.status_code != 200:
                raise ProviderError(
                    self.name,
                    f"API error: {response.status_code} - {response.text[:200]}",
                    retriable=response.status_code >= 500,
                    status_code=response.status_code,
                )
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return ProviderResponse(
                content=content,
                provider=self.name,
                model=self.model,
                tokens_used=usage.get("total_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_ms=(time.time() - start) * 1000,
                finish_reason=data["choices"][0].get("finish_reason", "stop"),
                raw_response=data,
            )
        except requests.exceptions.RequestException as e:
            raise ProviderError(
                self.name,
                f"Request failed: {e}",
                retriable=True,
            ) (duplicate-code)
---
src/workflow_composer/results/collector.py:1:0: R0801: Similar lines in 2 files
==workflow_composer.models.providers.lightning:[38:61]
==workflow_composer.models.providers.openai:[38:61]
        model = model or self.DEFAULT_MODEL
        messages = self._build_messages(prompt, system_prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text() (duplicate-code)
---
src/workflow_composer/results/collector.py:1:0: R0801: Similar lines in 2 files
==workflow_composer.models.utils.metrics:[131:174]
==workflow_composer.providers.utils.metrics:[147:190]
        session.total_latency_ms += latency_ms
        session.last_used = now
        if not success:
            session.errors += 1
        # Persist
        self._save_today()
    def get_daily_usage(self) -> Dict[str, ProviderUsage]:
        """Get today's usage by provider."""
        return dict(self._daily)
    def get_session_usage(self) -> Dict[str, ProviderUsage]:
        """Get this session's usage by provider."""
        return dict(self._session)
    def get_usage_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get usage summary for the last N days.
        Args:
            days: Number of days to summarize
        Returns:
            Summary with totals per provider
        """
        totals: Dict[str, ProviderUsage] = {}
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            file_path = self.metrics_dir / f"usage_{date}.json"
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                        for provider_id, usage in data.get("providers", {}).items():
                            if provider_id not in totals:
                                totals[provider_id] = ProviderUsage(
                                    provider_id=provider_id
                                )
 (duplicate-code)
---
src/workflow_composer/results/collector.py:1:0: R0801: Similar lines in 2 files
==workflow_composer.models.providers.gemini:[123:143]
==workflow_composer.models.providers.openai:[95:115]
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    latency = (time.time() - start) * 1000
                    if response.status == 200:
                        return {
                            "available": True,
                            "latency_ms": latency,
                        }
                    else:
                        return {
                            "available": False,
                            "error": f"HTTP {response.status}",
                            "latency_ms": latency,
                        }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
            } (duplicate-code)
---
src/workflow_composer/results/collector.py:1:0: R0801: Similar lines in 2 files
==workflow_composer.models.utils.metrics:[179:210]
==workflow_composer.providers.utils.metrics:[198:226]
                    pass
        return {
            "period_days": days,
            "providers": {
                provider_id: usage.to_dict()
                for provider_id, usage in totals.items()
            },
            "total_requests": sum(u.requests for u in totals.values()),
            "total_tokens": sum(u.tokens_used for u in totals.values()),
            "total_errors": sum(u.errors for u in totals.values()),
        }
    def estimate_costs(self, days: int = 30) -> Dict[str, float]:
        """
        Estimate costs based on token usage.
        Args:
            days: Number of days to calculate
        Returns:
            Estimated costs per provider in USD
        """
        # Approximate costs per 1M tokens
        COST_PER_MILLION = {
            "lightning": 0.0,   # Free tier
            "gemini": 0.0,      # Free tier
            "openai": 2.50,     # GPT-4o-mini average (duplicate-code)
---
src/workflow_composer/results/collector.py:1:0: R0801: Similar lines in 2 files
==workflow_composer.web.app:[31:46]
==workflow_composer.web.chat_handler:[82:98]
    try:
        result = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%i %j %T %N"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split('\n'):
            if line and ('vllm' in line.lower() or 'biopipelines' in line.lower()) and 'RUNNING' in line:
                parts = line.split()
                if len(parts) >= 4:
                    node = parts[3]
                    return f"http://{node}:8000/v1"
    except Exception:
        pass
    return os.environ.get("VLLM_URL", "http://localhost:8000/v1")
 (duplicate-code)
---
src/workflow_composer/results/collector.py:1:0: R0801: Similar lines in 2 files
==workflow_composer.providers.openai:[117:135]
==workflow_composer.providers.vllm:[106:124]
            )
            if response.status_code != 200:
                raise ProviderError(
                    self.name,
                    f"API error: {response.status_code} - {response.text[:200]}",
                    retriable=response.status_code >= 500,
                    status_code=response.status_code,
                )
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return ProviderResponse(
                content=content,
                provider=self.name, (duplicate-code)
---
... and 5 more duplicate groups
```

**Total duplicate groups:** 15

## 💀 Dead Code Detection (Vulture)

Vulture finds unused code with confidence scores.
Higher confidence = more likely truly unused.

```
src/workflow_composer/agents/autonomous/agent.py:324: unreachable code after 'return' (100% confidence)
src/workflow_composer/agents/autonomous/health_checker.py:25: unused import 'socket' (90% confidence)
src/workflow_composer/agents/chat_integration.py:19: unused import 'lru_cache' (90% confidence)
src/workflow_composer/agents/enhanced_tools.py:613: unused variable 'log_type' (100% confidence)
src/workflow_composer/agents/executor/file_ops.py:535: unused variable 'diff_text' (100% confidence)
src/workflow_composer/agents/executor/process_manager.py:36: unused import 'signal' (90% confidence)
src/workflow_composer/agents/tools.py:42: unused import 'DataDiscoveryTools' (90% confidence)
src/workflow_composer/agents/tools.py:42: unused import 'DataManagementTools' (90% confidence)
src/workflow_composer/agents/tools.py:42: unused import 'MODULAR_TOOL_PATTERNS' (90% confidence)
src/workflow_composer/agents/tools.py:42: unused import 'ModularAgentTools' (90% confidence)
src/workflow_composer/agents/tools.py:42: unused import 'WorkflowTools' (90% confidence)
src/workflow_composer/agents/tools.py:1907: unused variable 'comparison' (100% confidence)
src/workflow_composer/agents/tools/diagnostics.py:211: unused variable 'result_type' (100% confidence)
src/workflow_composer/agents/tools/education.py:327: unused variable 'comparison_type' (100% confidence)
src/workflow_composer/composer.py:42: unused import 'TYPE_CHECKING' (90% confidence)
src/workflow_composer/data/downloader.py:28: unused import 'urlparse' (90% confidence)
src/workflow_composer/diagnosis/gemini_adapter.py:87: unused import 'google' (90% confidence)
src/workflow_composer/llm/huggingface_adapter.py:282: unused import 'transformers' (90% confidence)
src/workflow_composer/results/cloud_transfer.py:175: unused import 'ClientError' (90% confidence)
src/workflow_composer/web/archive/api.py:26: unused import 'BackgroundTasks' (90% confidence)
src/workflow_composer/web/archive/app.py:21: unused import 'render_template' (90% confidence)
src/workflow_composer/web/archive/unified_workspace.py:137: unused import 'PipelineExecutor' (90% confidence)
src/workflow_composer/web/archive/unified_workspace.py:400: unused variable 'current_input' (100% confidence)
src/workflow_composer/web/archive/unified_workspace.py:405: unused variable 'current_input' (100% confidence)
src/workflow_composer/web/components/autonomous_panel.py:34: unused import 'ExecAutonomyLevel' (90% confidence)
```

**Total issues:** 25

## 📊 Code Complexity Analysis (Radon)

Complexity grades: A (simple) → F (very complex)
Focus on C, D, E, F rated functions for refactoring.

### High Complexity Functions (C or worse)
```
src/workflow_composer/cli.py
  F 65:0 cmd_chat - D (23)
  F 252:0 cmd_ui - C (15)
src/workflow_composer/composer.py
  M 364:4 Composer.check_readiness - E (34)
  M 148:4 Composer.generate - C (19)
  M 261:4 Composer._enhance_intent_from_manifest - C (19)
  M 478:4 Composer.validate_and_prepare - C (11)
src/workflow_composer/models/router.py
  M 123:4 ModelRouter.complete_async - C (11)
src/workflow_composer/core/preflight_validator.py
  M 259:4 PreflightValidator.validate - C (17)
  M 109:4 ValidationReport.to_markdown - C (12)
src/workflow_composer/core/model_service_manager.py
  M 111:4 ModelOrchestrator._check_biomistral_status - C (13)
src/workflow_composer/core/query_parser_ensemble.py
  M 584:4 EnsembleIntentParser._combine_votes - D (23)
src/workflow_composer/core/workflow_generator.py
  M 456:4 WorkflowGenerator._generate_from_file_template - C (11)
  M 523:4 WorkflowGenerator._generate_from_template - C (11)
src/workflow_composer/core/query_parser.py
  M 451:4 IntentParser._parse_with_rules - D (24)
src/workflow_composer/core/module_mapper.py
  M 421:4 ModuleMapper._guess_category - C (15)
src/workflow_composer/web/chat_handler.py
  M 629:4 UnifiedChatHandler.chat - C (20)
  M 519:4 UnifiedChatHandler._map_arguments - C (11)
src/workflow_composer/web/components/data_tab.py
  F 355:0 search_databases_handler - D (24)
  F 169:0 scan_directory_handler - C (18)
src/workflow_composer/web/archive/api.py
  F 340:0 generate_workflow - C (11)
src/workflow_composer/web/archive/app.py
  F 609:0 generate_mock_workflow - C (12)
src/workflow_composer/diagnosis/agent.py
  M 447:4 ErrorDiagnosisAgent._match_patterns - C (13)
  M 269:4 ErrorDiagnosisAgent.diagnose - C (12)
  M 361:4 ErrorDiagnosisAgent.diagnose_from_logs_with_llm - C (12)
  M 557:4 ErrorDiagnosisAgent._get_available_llm - C (12)
src/workflow_composer/diagnosis/auto_fix.py
  M 173:4 AutoFixEngine.execute - C (13)
src/workflow_composer/diagnosis/log_collector.py
  M 121:4 LogCollector.collect - E (32)
  M 35:4 CollectedLogs.get_combined_error_context - C (16)
src/workflow_composer/diagnosis/github_agent.py
  F 760:0 get_github_copilot_agent - C (12)
src/workflow_composer/agents/tools.py
  M 1976:4 AgentTools.describe_files - F (64)
  M 2251:4 AgentTools.validate_dataset - F (64)
  M 1491:4 AgentTools.monitor_jobs - F (43)
  M 576:4 AgentTools.cleanup_data - E (33)
  M 851:4 AgentTools.search_databases - E (33)
  M 2953:4 AgentTools.analyze_results - D (27)
  M 468:4 AgentTools.scan_data - D (23)
  M 2592:4 AgentTools.search_tcga - C (20)
  C 319:0 AgentTools - C (16)
  M 761:4 AgentTools._get_available_partition - C (15)
  M 1019:4 AgentTools.download_dataset - C (15)
  M 1220:4 AgentTools.check_references - C (15)
  M 1700:4 AgentTools.diagnose_error - C (15)
  M 381:4 AgentTools.detect_tool - C (11)
  M 1341:4 AgentTools.submit_job - C (11)
  M 1907:4 AgentTools.compare_samples - C (11)
src/workflow_composer/agents/react_agent.py
  M 141:4 ReactAgent.run - D (21)
src/workflow_composer/agents/router.py
  M 511:4 AgentRouter._route_with_regex - D (24)
src/workflow_composer/agents/validation.py
  M 372:4 ResponseValidator.validate_scan_result - D (24)
  M 71:4 ConversationContext.add_message - C (17)
  M 235:4 ConversationContext.matches_intent - C (14)
  M 113:4 ConversationContext._extract_intent - C (13)
  M 581:4 ResponseValidator._build_validated_response - C (13)
  C 58:0 ConversationContext - C (12)
  C 360:0 ResponseValidator - C (12)
  M 491:4 ResponseValidator.validate_search_result - C (12)
  M 540:4 ResponseValidator._analyze_sample_names - C (12)
src/workflow_composer/agents/bridge.py
  M 161:4 AgentBridge._args_dict_to_list - C (17)
src/workflow_composer/agents/chat_integration.py
  M 279:4 AgentChatHandler._chat_sync - D (28)
  M 448:4 AgentChatHandler._chat_async - C (11)
src/workflow_composer/agents/enhanced_tools.py
  M 1156:4 SystemHealthTool.execute - D (22)
  C 1146:0 SystemHealthTool - C (13)
src/workflow_composer/agents/self_healing.py
  M 203:4 SelfHealer._get_job_info - C (12)
  M 410:4 SelfHealer._map_diagnosis_to_action - C (12)
src/workflow_composer/agents/autonomous/recovery.py
  M 211:4 RecoveryManager.handle_job_failure - C (16)
src/workflow_composer/agents/autonomous/agent.py
  M 925:4 AutonomousAgent._direct_tool_execution - E (31)
  M 420:4 AutonomousAgent._execute_action_internal - C (16)
src/workflow_composer/agents/autonomous/health_checker.py
  M 240:4 HealthChecker.check_gpu - C (14)
  M 460:4 HealthChecker.check_slurm - C (12)
src/workflow_composer/agents/autonomous/job_monitor.py
  M 316:4 JobMonitor._check_job - C (13)
src/workflow_composer/agents/tools/base.py
  M 189:4 RegisteredTool.validate_arguments - C (17)
src/workflow_composer/agents/tools/diagnostics.py
  F 209:0 analyze_results_impl - C (18)
  F 27:0 diagnose_error_impl - C (14)
src/workflow_composer/agents/tools/data_management.py
  F 124:0 cleanup_data_impl - D (21)
src/workflow_composer/agents/tools/data_discovery.py
  F 40:0 scan_data_impl - D (23)
  F 160:0 search_databases_impl - D (23)
  F 443:0 describe_files_impl - D (22)
  F 547:0 validate_dataset_impl - C (15)
src/workflow_composer/agents/executor/sandbox.py
  M 262:4 CommandSandbox.execute - C (13)
src/workflow_composer/agents/executor/audit.py
  M 356:4 AuditLogger.get_entries - C (16)
src/workflow_composer/agents/executor/permissions.py
  M 507:4 PermissionManager._matches - C (12)
src/workflow_composer/providers/router.py
  M 237:4 ProviderRouter.chat - C (13)
src/workflow_composer/providers/anthropic.py
  M 61:4 AnthropicProvider.chat - C (11)
src/workflow_composer/providers/utils/health.py
  F 131:0 print_health_report - C (13)
src/workflow_composer/data/scanner.py
  M 328:4 LocalSampleScanner._order_pair - D (21)
  M 226:4 LocalSampleScanner._match_pairs - C (15)
src/workflow_composer/data/reference_manager.py
  M 215:4 ReferenceManager.check_references - C (14)
src/workflow_composer/data/manifest.py
  M 523:4 DataManifest.to_samplesheet - C (17)
  M 317:4 ReferenceInfo.get_available_indexes - C (13)
  M 655:4 DataManifest.summary - C (11)
src/workflow_composer/data/discovery/orchestrator.py
  M 274:4 DataDiscovery._rank_results - C (20)
  M 93:4 DataDiscovery.search - C (12)
src/workflow_composer/data/discovery/query_parser.py
  M 225:4 QueryParser._parse_with_rules - D (24)
src/workflow_composer/data/discovery/adapters/geo.py
  M 419:4 GEOAdapter._get_sra_runs_for_gse - C (11)
src/workflow_composer/data/discovery/adapters/ensembl.py
  M 74:4 EnsemblAdapter.search - D (21)
  M 335:4 EnsemblAdapter.validate_local_reference - C (20)
src/workflow_composer/data/discovery/adapters/encode.py
  M 280:4 ENCODEAdapter._build_search_params - C (19)
src/workflow_composer/data/browser/reference_browser.py
  F 108:0 get_dataset_details - C (15)
  F 51:0 search_datasets - C (14)
  F 203:0 scan_local_references - C (14)
src/workflow_composer/monitor/workflow_monitor.py
  M 152:4 WorkflowMonitor._parse_nextflow_log - C (11)
  M 313:4 WorkflowMonitor.generate_report - C (11)
src/workflow_composer/results/archiver.py
  M 215:4 ResultArchiver._filter_files - C (16)
src/workflow_composer/results/collector.py
  M 245:4 ResultCollector._build_summary - C (12)
  M 112:4 ResultCollector._walk_directory - C (11)
```

**Average complexity: A (4.014891179839633)**

## 🔧 Maintainability Index (Radon)

MI grades: A (highly maintainable) → C (difficult to maintain)

### Files with Low Maintainability (C rated)
```
src/workflow_composer/agents/tools.py - C (0.00)
```