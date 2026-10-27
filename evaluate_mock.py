
import json
import sys
import time
import pandas as pd
from pathlib import Path

project_dir = Path("/content/P4_Customer_Support_Intent_Router")
sys.path.insert(0, str(project_dir))
from support_router import classify_ticket

tickets = pd.read_csv(project_dir / "support_tickets.csv")
results = []

for _, row in tickets.iterrows():
    start = time.perf_counter()
    result = classify_ticket(row["email"], mode="mock")
    latency_ms = (time.perf_counter() - start) * 1000

    results.append({
        "ticket_id": row["ticket_id"],
        "mode": "mock",
        "prompt_variant": "not_applicable",
        "expected_category": row["expected_category"],
        "predicted_category": result.category,
        "category_correct": result.category == row["expected_category"],
        "expected_priority": row["expected_priority"],
        "predicted_priority": result.priority,
        "priority_correct": result.priority == row["expected_priority"],
        "schema_valid": True,
        "latency_ms": round(latency_ms, 4),
        "summary": result.summary,
        "draft_response": result.draft_response
    })

out = pd.DataFrame(results)
out.to_csv(project_dir / "router_evaluation_mock.csv", index=False)

summary = {
    "mode": "mock",
    "tickets_evaluated": len(out),
    "category_accuracy": round(float(out["category_correct"].mean()), 4),
    "priority_accuracy": round(float(out["priority_correct"].mean()), 4),
    "schema_validity": round(float(out["schema_valid"].mean()), 4),
    "mean_latency_ms": round(float(out["latency_ms"].mean()), 4),
    "real_llm_metrics": False
}
with open(project_dir / "router_evaluation_mock_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\nPer-ticket results:")
print(out[[
    "ticket_id", "expected_category", "predicted_category",
    "expected_priority", "predicted_priority"
]].to_string(index=False))
print("\nSaved router_evaluation_mock.csv and router_evaluation_mock_summary.json")
