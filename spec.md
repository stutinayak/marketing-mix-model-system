# Marketing Mix Model System - Specification

## Overview
Design and implement a prototype system for Marketing Mix Modeling (MMM) that demonstrates strong programming, system design, and data processing skills.

## Functional Requirements

### 1. Data Processing and Storage
- Ingest marketing spend and sales data from CSV files (one year of daily data).
- Implement data validation and cleaning mechanisms.
- Ensure efficient data storage and retrieval.

### 2. Model Service
- Implement a service to train and serve a Marketing Mix Model (MMM).
- Design the service to be modular for easy swapping of model types in the future.
- Optimize for performance.

### 3. API Layer
- Create a RESTful API using FastAPI.
- API should accept marketing spend data and return sales predictions.

## Dataset
- **sales_data.csv**: Daily sales figures
- **tv_spend.csv**: Daily TV advertising spend
- **radio_spend.csv**: Daily radio advertising spend
- **social_media_spend.csv**: Daily social media advertising spend
- **search_spend.csv**: Daily search engine advertising spend
- **print_spend.csv**: Daily print advertising spend
- **outdoor_spend.csv**: Daily outdoor advertising spend

## Deliverables
- A well-structured Python project.
- `README.md` with:
  - Setup instructions
  - System architecture overview
  - Discussion on extensibility and future improvements

## Evaluation Criteria
- Programming skills and problem-solving ability
- System design and architecture
- Code quality, readability, and best practices
- Use of design patterns and SOLID principles
- Data processing implementation
- Model implementation and validation
- API design and implementation
- Clear code documentation and README

## Time Expectation
- Designed to be completed in approximately one working day (focus on core skills, not exhaustive completeness). 