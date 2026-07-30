# 🎓 Interview Preparation - Ticket Classifier

Master talking points for your interviews. **Keep answers concise (30-60 seconds).**

---

## Part 1: Architecture & Design

### Q: Walk me through your ticket classifier architecture.

**Answer (30 seconds):**
> The system has three stages:
> 1. **Preprocessing**: Clean text (lowercase, remove punctuation/stopwords)
> 2. **Vectorization**: Convert text to 1000 TF-IDF features
> 3. **Classification**: Multinomial Naive Bayes predicts category + confidence
> 4. **Quality Control**: Flag low-confidence predictions for human review
> 
> Architecture is simple, fast, and explainable.

---

### Q: Why TF-IDF + Naive Bayes over deep learning?

**Answer (30 seconds):**
> Three reasons:
> 1. **Data efficiency**: Only 400 training samples → neural nets would overfit
> 2. **Speed**: TF-IDF training is <1 second, inference is ~3ms
> 3. **Explainability**: Easy to explain predicted category to stakeholders
> 
> Deep learning shines with 100k+ samples and GPUs. This solution is appropriate for the problem.

---

### Q: How do you handle edge cases?

**Answer (45 seconds):**
> Three strategies:
> 
> 1. **Ambiguous tickets** (e.g., "Help"): 
>    - Low confidence → flagged for human review
> 
> 2. **Mixed-category tickets** (e.g., login + billing issue):
>    - Confidence split across multiple classes
>    - If confidence < 60% → human review
> 
> 3. **Out-of-vocabulary words** (misspellings, jargon):
>    - TF-IDF handles gracefully (treats as low-frequency words)
>    - Still produces predictions
> 
> Key insight: **Confidence threshold is quality control**, not just model accuracy.

---

## Part 2: Model Training & Evaluation

### Q: What's your train/test split? Why that ratio?

**Answer (20 seconds):**
> 80/20 split: 320 training, 80 test samples.
> 
> I used **stratified split** to maintain class distribution (100 of each category stays 80/20 in train/test).
> 
> 80/20 is industry standard → maximizes training data while preserving test integrity.

---

### Q: How do you evaluate model performance?

**Answer (45 seconds):**
> Four layers of metrics:
> 
> 1. **Accuracy** (98.75%): Overall correctness
> 2. **Precision**: False positives → misclassified tickets
> 3. **Recall**: False negatives → missed tickets
> 4. **Confusion Matrix**: Shows which categories are confused
> 
> Example: "Billing has 100% precision but 90% recall"
> = No false positives (good!) but misses 10% of actual billing tickets (bad)
> 
> This tells me to investigate why billing tickets are missed.

---

### Q: What's the confusion matrix telling you?

**Answer (30 seconds):**
> The diagonal shows correct predictions (high values = good).
> Off-diagonal shows misclassifications.
> 
> Example: If a Technical ticket is misclassified as Billing 2 times:
> - Suggests these categories share vocabulary
> - Could indicate ambiguous tickets
> - Might warrant additional training data or category refinement

---

### Q: How do you prevent overfitting with 400 samples?

**Answer (40 seconds):**
> Three mechanisms:
> 
> 1. **Simple model**: Naive Bayes has few parameters vs neural networks
> 2. **TF-IDF limits**: Max 1000 features prevents dimension explosion
> 3. **Test validation**: 98.75% accuracy on train ≈ 98% on test (similar) → no overfitting
> 
> If train accuracy was 99% but test was 85%, I'd see overfitting and investigate.

---

## Part 3: Confidence Scoring & Production

### Q: How do you use confidence scores in production?

**Answer (45 seconds):**
> Confidence scores enable **risk-based routing**:
> 
> - **High confidence (≥60%)**: Auto-assign to category
>   - Fast resolution (seconds)
>   - User gets instant response
> 
> - **Low confidence (<60%)**: Flag for human review
>   - Prevents costly mistakes
>   - Human agent double-checks
>   - Creates feedback loop
> 
> This threshold is tunable based on business requirements:
> - Higher threshold = safer but slower
> - Lower threshold = faster but riskier

---

### Q: How would you handle data drift?

**Answer (45 seconds):**
> Data drift = model performance degrades over time.
> 
> **Detection**:
> ```python
> if production_accuracy < test_accuracy - 0.05:  # 5% drop
>     alert_retraining()
> ```
> 
> **Prevention**:
> 1. Monitor confidence distribution weekly
> 2. Retrain on new tickets monthly
> 3. Track human review feedback → use as new training data
> 4. Version models → rollback if needed
> 
> For this project, I'd monitor if "needs review" rate exceeds 15%.

---

### Q: Can you deploy this to production?

**Answer (45 seconds):**
> Yes, multiple options:
> 
> 1. **Streamlit Cloud** (easiest, my current setup):
>    - Push to GitHub → auto-deploys
>    - Free tier handles ~1000 concurrent users
> 
> 2. **Docker + Cloud Run/Heroku**:
>    - Containerized, scalable to 100k+ requests/sec
> 
> 3. **Flask/FastAPI microservice**:
>    - REST API for ticketing system integration
>    - Load balancer for horizontal scaling
> 
> Model is lightweight (~1MB) → no GPU needed.

---

## Part 4: Improvements & Scaling

### Q: How would you improve accuracy?

**Answer (45 seconds):**
> Three tiers:
> 
> **Tier 1 (Easy)**:
> - Expand data: 400 → 2000+ tickets (10x more patterns)
> - Tune threshold based on business metrics
> 
> **Tier 2 (Medium)**:
> - Add metadata: priority, urgency, sender type
> - Try SVM or XGBoost (may gain 1-2%)
> - Ensemble multiple models
> 
> **Tier 3 (Hard)**:
> - Fine-tune BERT embeddings (needs GPU, more complex)
> - Active learning: sample uncertain predictions → annotate → retrain
> 
> But current 98.75% is excellent → *ship first, optimize later*.

---

### Q: How would you scale to 100k daily tickets?

**Answer (45 seconds):**
> Current bottleneck: Streamlit (single-threaded).
> 
> Scaling strategy:
> 
> 1. **Replace Streamlit** with Flask/FastAPI
> 2. **Load model once** on server startup (not per request)
> 3. **Batch processing**: Group 100 tickets → single inference
> 4. **Caching**: Popular tickets cached
> 5. **Async queues**: Kafka for high-volume ingestion
> 6. **Horizontal scaling**: Multiple replicas behind load balancer
> 
> Current setup handles ~1000 tickets/sec. For 100k/day = 1.2/sec → plenty of headroom.

---

## Part 5: Tricky Questions

### Q: What's the difference between precision and recall?

**Answer (30 seconds):**
> **Precision**: Of tickets we labeled as "Billing", how many were actually billing?
> - High precision = few false positives
> - Important when wrong classification is costly
> 
> **Recall**: Of actual billing tickets, how many did we find?
> - High recall = few false negatives
> - Important when missing tickets is costly
> 
> Example: Medical diagnosis system needs high recall (don't miss disease). Spam filter needs high precision (don't flag good emails).

---

### Q: Why not just use 100% confidence threshold?

**Answer (20 seconds):**
> Because no prediction is 100% certain.
> 
> If we required 100% confidence:
> - 0 tickets would auto-assign
> - 100% need human review
> - System is useless
> 
> Threshold (60%) is a sweet spot: most tickets auto-assign, ambiguous ones get reviewed.

---

### Q: How do you choose the confidence threshold?

**Answer (45 seconds):**
> **Data-driven approach**:
> 
> 1. Train model, get confidence distribution
> 2. Plot: threshold vs (auto-rate, error-rate)
> 3. Choose threshold where trade-off makes sense
> 
> **Business considerations**:
> - If auto-assignment is mission-critical → high threshold (90%)
> - If resolution speed matters → low threshold (40%)
> - Default (60%) balances speed and safety
> 
> For this project, 60% means ~95% auto-assign, ~5% review → good balance.

---

### Q: What if a new category arrives?

**Answer (30 seconds):**
> Current model only knows 4 categories (Billing, Technical, HR, General).
> 
> If a "Refund Fraud" category emerges:
> 1. Collect 100+ examples
> 2. Retrain model (includes new category)
> 3. Deploy new version
> 4. Monitor accuracy on new category
> 
> Takes ~2 hours end-to-end. Could automate with CI/CD pipeline.

---

## Part 6: Show & Tell

### Q: Can you show me the model code?

**Answer (5 minutes):**
Show them the training script:

```python
# Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))

# Inference with confidence
category = model.predict(X_vec)[0]
confidence = model.predict_proba(X_vec).max()
status = "Auto Assigned" if confidence >= 0.60 else "Needs Review"
```

Key points to highlight:
- Simple, readable code
- Comprehensive evaluation
- Confidence scoring baked in

---

### Q: What does TF-IDF actually compute?

**Answer (45 seconds):**
> TF-IDF = Term Frequency × Inverse Document Frequency
> 
> **TF** (Term Frequency):
> - How often does word appear in this ticket?
> - "payment" appears 3 times → TF = 3
> 
> **IDF** (Inverse Document Frequency):
> - How rare is this word across all tickets?
> - "payment" appears in 80% of docs → IDF = log(1/0.8) = low
> - "invoice" appears in 5% of docs → IDF = log(1/0.05) = high
> 
> **Result**: TF-IDF scores common words low, rare words high.
> Example: "payment" might get 0.3, "invoice" might get 0.8 → invoice is more meaningful.

---

## Part 7: Common Mistakes to Avoid

### ❌ Don't Say:
- "I achieved 98% accuracy" (without mentioning precision/recall)
- "I used deep learning" (overkill for 400 samples)
- "I had no false positives" (suspicious, usually sign of overfitting)
- "Confidence means the prediction is correct" (no, it's a probability)

### ✅ Do Say:
- "98.75% accuracy with 1.0 precision on Billing, 0.9 recall"
- "TF-IDF + Naive Bayes is appropriate for this data size and deployment constraints"
- "Confusion matrix shows 2 misclassifications between Billing and Technical"
- "Confidence is the model's uncertainty; >60% = auto-assign, <60% = human review"

---

## Part 8: Red Flags to Avoid

### 🚩 Don't Get Caught Here

**"What's your model accuracy?"**
- ❌ "98.75%"
- ✅ "98.75% overall, but Billing precision is 100%, recall is 90% — let me show you the confusion matrix"

**"Why didn't you use neural networks?"**
- ❌ "They're too complex"
- ✅ "With 400 training samples, neural networks would overfit. TF-IDF + Naive Bayes trains in 100ms and achieves 98% accuracy."

**"How do you handle out-of-distribution data?"**
- ❌ "I don't"
- ✅ "Confidence scoring identifies uncertain predictions. Out-of-distribution tickets get low confidence and route to human review."

---

## Part 9: Practice Questions (Test Yourself)

### Easy
1. What's the training accuracy vs test accuracy? (Answer: Similar → no overfitting)
2. What categories does your model handle? (Answer: Billing, Technical, HR, General)
3. What's your confidence threshold? (Answer: 60%)

### Medium
1. How would you detect if the model is degrading? (Answer: Monitor confidence distribution)
2. What's the inference latency? (Answer: ~3-4ms per ticket)
3. If recall for Billing drops, what would you investigate? (Answer: More billing data, threshold tuning, feature analysis)

### Hard
1. How do you balance precision vs recall? (Answer: Business requirements; different stakeholders have different needs)
2. Explain your train/test split strategy (Answer: Stratified 80/20 maintains class distribution)
3. How would you handle imbalanced data? (Answer: Stratified split, class weights, F1-score instead of accuracy)

---

## Part 10: The Closer

When interviewer asks: **"Any questions for me?"**

Ask something that shows you think about production:
- "What's your volume? Would 100k/day require a different architecture?"
- "How do you currently route low-confidence predictions to humans?"
- "Do you have feedback loops to catch misclassifications?"
- "What are the cost implications of false positives vs false negatives?"

This shows you think beyond the model → production impact.

---

## Final Checklist Before Interview

- [ ] Explain architecture in <1 minute
- [ ] Know accuracy, precision, recall numbers
- [ ] Understand confusion matrix
- [ ] Be ready to defend TF-IDF + Naive Bayes choice
- [ ] Know confidence threshold rationale
- [ ] Have deployment strategy ready
- [ ] Practice talking about edge cases
- [ ] Prepare for "how would you improve" question
- [ ] Be humble about limitations (400 samples, single language, etc.)

---

## Example Full Response (Timed)

**"Tell me about your ticket classifier project"** *(1 minute)*

> I built an end-to-end ML system that classifies support tickets into 4 categories with 98.75% accuracy.
> 
> **Architecture**: Text preprocessing → TF-IDF vectorization → Multinomial Naive Bayes → confidence scoring.
> 
> **Key innovation**: Instead of just predicting categories, I output a confidence score. Predictions >60% confidence auto-assign instantly. Below 60% get flagged for human review. This quality control approach is more important than accuracy alone.
> 
> **Evaluation**: 98.75% overall accuracy, high precision (100%) and recall (90%+) across categories. Confusion matrix shows no systematic errors.
> 
> **Production-ready**: Deployed via Streamlit, handles 1000+ requests/sec, model inference is 3-4ms. Can scale horizontally to 100k daily tickets if needed.
> 
> **Why this approach**: TF-IDF + Naive Bayes is simple, fast, and explainable. Deep learning would overfit on 400 samples. This system is designed for practical deployment, not just benchmark numbers.
> 
> Questions?

---

## Resources

- **Scikit-learn docs**: https://scikit-learn.org/stable/modules/naive_bayes.html
- **TF-IDF explanation**: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- **Precision/Recall**: https://en.wikipedia.org/wiki/Precision_and_recall
- **Confusion matrix**: https://en.wikipedia.org/wiki/Confusion_matrix

---

**Good luck! You've got this! 🚀**
