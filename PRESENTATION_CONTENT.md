# ⚡ TechSprint2: Manipur PowerGuard - Detailed Project Showcase

## 📅 Slide 1: The Challenge - Securing the "Last Mile" of Power Distribution
**Title:** "Combating Non-Technical Losses in Challenging Geographies"

### **The Context ($16 Billion Problem)**
*   **The Crisis:** Electricity distribution companies (DISCOMs) in India face massive **Aggregate Technical & Commercial (AT&C) losses**, often exceeding 20-30%.
*   **The Specific Problem:** "Non-Technical Loss" (Theft) is the biggest controllable factor.
    *   **Direct Hooking:** Tapping overhead lines (common in rural areas).
    *   **Meter Tampering:** Bypassing meters or slowing them down (common in urban/commercial).
    *   **Grid Instability:** Unaccounted load causes transformer blowouts and voltage fluctuations.

### **Why It's Hard to Solve in Manipur**
1.  **Geography:** Hilly terrain in districts like Churachandpur/Ukhrul makes physical manual inspection difficult and expensive.
2.  **Manpower Shortage:** Impossible to manually review millions of meter readings daily.
3.  **Sophisticated Theft:** Modern theft isn't just "0" usage; it's subtle (e.g., using partial loads or bypassing only at night) which standard rules miss.

---

## 💡 Slide 2: The Solution - "Manipur GridWatch" AI Engine
**Title:** "Hybrid Intelligence: Combining Statistical Rules with ML"

### **Core Philosophy: Precision over Recall**
From a business perspective, sending a truck to inspect a home costs money. Our system prioritizes **High Precision (87.5%)** to ensure that when we flag a consumer, it is almost certainly theft, minimizing wasted field trips.

### **The "Three-Tier" Defense Architecture**
1.  **Tier 1: Statistical Rules (The "Obvious"):**
    *   Flags impossible values (Zero consumption for active accounts, usage > sanctioned load).
2.  **Tier 2: Unsupervised Learning (The "Unknowns"):**
    *   **Algorithm:** *Isolation Forest*
    *   **Role:** Detects anomalies we haven't seen before—outliers that statistically deviate from the grid's normal variance.
3.  **Tier 3: Supervised Learning (The "Known Patterns"):**
    *   **Algorithm:** *Random Forest Classifier*
    *   **Role:** Trained on specific theft signatures (Sudden Drops, erratic spikes, pattern deviations) to classify intent.

---

## 🛠️ Slide 3: Architecture & Data Flow
**Title:** "Under the Hood: How Manipur GridWatch Works"

### **1. The System Architecture & Flow**
`[Smart Meter Data] ➔ [Preprocessing] ➔ [Feature Engine] ➔ [Hybrid AI Core] ➔ [Dashboard]`

*   **Step 1: Ingestion & Preprocessing:**
    *   Ingests raw Time/kWh logs.
    *   Handles missing timestamps and normalizes scale.
*   **Step 2: Advanced Feature Engineering (The Secret Sauce):**
    *   Transforms raw data into **44 behavioral signals**:
    *   *Rolling Stats:* 24h/72h Averages (catch shifts).
    *   *Lag Features:* "Same time yesterday" comparison.
    *   *CoV:* Coefficient of Variation (Theft is often erratic).
*   **Step 3: The Hybrid AI Core:**
    *   **Tier 1 (Unsupervised):** *Isolation Forest* detects unknown anomalies.
    *   **Tier 2 (Supervised):** *Random Forest* confirms known theft signatures.
    *   **Voting Mechanism:** Weighted consensus to reduce false alarms.
*   **Step 4: Visualization Layer:**
    *   Delivers "High Probability" flags to the Streamlit Dashboard.

### **2. The Tech Stack**
*   **Python 3.13 & Pandas:** Optimized for speed on CPU.
*   **Scikit-Learn:** Chosen for explainability and easy deployment on utility servers (No GPUs needed).
*   **Streamlit & Plotly:** For rapid, interactive geospatial visualization.

---

## 🚀 Slide 4: Hackathon Achievements
**Title:** "Mission Accomplished: Phase 1 Deliverables"

### **1. What We Built (The Hackathon Scope)**
We have successfully developed and deployed the **Core MVP** for the 48-hour challenge:
*   **Fully Functional AI Engine:** Trained on 1,000 Manipur-specific profiles.
*   **Operational Dashboard:** Live tracking of revenue loss and geolocation.
*   **Localized Data:** Custom generator creating 8-district topography authentic to Manipur.

### **2. Validation Metrics (Phase 1 Results)**
*   **Precision: 87.5%:** (Key Business Metric: Low False Positives).
*   **ROC-AUC: 0.926:** Excellent model discrimination capability.
*   **Latency:** Processes 24 hours of data for a district in <5 seconds.

### **3. Live Dashboard Features**
*   **Revenue at Risk Calculator:** Real-time financial estimator.
*   **District Heatmaps:** Operational view for regional managers.
*   **Consumer Forensics:** Drill-down view for individual investigation.

---

## 🔮 Slide 5: Roadmap & Future Development
**Title:** "From Prototype to State-Wide Deployment"

### **Phase 1: Integration (Months 1-3)**
*   **API Hook:** Connect directly to Smart Meter (AMI) Head End Systems (HES).
*   **Feedback Loop:** "Human-in-the-loop" button on dashboard for inspectors to mark "Confirmed Theft" or "False Alarm," automatically retraining the model.

### **Phase 2: Advanced Capabilities (Months 4-6)**
*   **Deep Learning (LSTM/GRU):** For better forecasting of "expected" load to detect subtler deviations.
*   **Edge AI:** Deploy lightweight TFLite models directly onto **Data Concentrator Units (DCUs)** to detect theft even if network connectivity is lost.

### **Phase 3: Grid Ecology**
*   **Predictive Maintenance:** Use the same data to predict transformer overloads before they happen.
*   **Consumer App:** Send "High Usage Alerts" to consumers to prevent bill shock (turning a policing tool into a customer service tool).
**Title:** "MVP Status: Operational & Localized"

### **1. Real-World Simulation (Manipur Edition)**
We didn't just use generic data. We built a custom **Data Generator** for the Hackathon:
*   **1,000 Consumers:** Modeled on real Manipur demographics (authentic names, community distribution).
*   **8 Districts:** Geospatial mapping of Imphal East/West, Thoubal, Churachandpur, etc.
*   **Realistic Anomalies:** Injected 5 distinct theft types (Spikes, Gradual reductions, Zero usage, etc.).

### **2. Performance Metrics**
*   **Precision: 87.5%:** (Key Business Metric: Low False Positives).
*   **ROC-AUC: 0.926:** Excellent model discrimination capability.
*   **Latency:** Processes 24 hours of data for a district in <5 seconds.

### **3. Live Dashboard Features**
*   **Revenue at Risk Calculator:** Real-time financial estimator.
*   **District Heatmaps:** Operational view for regional managers.
*   **Consumer Forensics:** Drill-down view for individual investigation.

---

## 🔮 Slide 5: Roadmap & Future Development
**Title:** "From Prototype to State-Wide Deployment"

### **Phase 1: Integration (Months 1-3)**
*   **API Hook:** Connect directly to Smart Meter (AMI) Head End Systems (HES).
*   **Feedback Loop:** "Human-in-the-loop" button on dashboard for inspectors to mark "Confirmed Theft" or "False Alarm," automatically retraining the model.

### **Phase 2: Advanced Capabilities (Months 4-6)**
*   **Deep Learning (LSTM/GRU):** For better forecasting of "expected" load to detect subtler deviations.
*   **Edge AI:** Deploy lightweight TFLite models directly onto **Data Concentrator Units (DCUs)** to detect theft even if network connectivity is lost.

### **Phase 3: Grid Ecology**
*   **Predictive Maintenance:** Use the same data to predict transformer overloads before they happen.
*   **Consumer App:** Send "High Usage Alerts" to consumers to prevent bill shock (turning a policing tool into a customer service tool).
