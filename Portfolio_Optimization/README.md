# Portfolio optimization model
This project is dedicated to building the optimization model of securities portfolio. The model contains 4 classes.
1. class Portfolio - initialize portfolio, upload data, convert data for suitable work and make some calculations such as portfolio return or risk.
2. class OptimizationP(Portfolio) - solve optimization tasks, make calculation for building efficient frontier and not efficient region.
3. class Model - model for simulation, create inital allocation and make some actions for forecasting and computing return.
4. class PortfolioModel(OptimizationP) - the main class which display efficient frontier with solved tasks and evaluate indicators which characterize portfolios.
