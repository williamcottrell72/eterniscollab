# Eternis Betting Markets

### Generating Distributions

We need the ability to estimate probabiliities from scratch in order to initialize a market.  Additionally - it would be useful to have various metrics on a given topic for future modeling purposes.  To that end I've added "interest" and "divisiveness".  (We should probably also add "investability" - i.e., how much does the outcome of the market effect tradeable equities?)

### Generating New Markets

Considerations:
  1. To generate engagement we want popular, divisive topics.  How well are we able to predict which topics will have the desired effect?
  2. There are really two kinds of divisive topics, 1) those for which people feel strongly and 2) in the literal sense that the topics divide the population into two even groups.  We should pay special attention to both of these flavors of divisive.  If people feel strongly (or make large bets) then we can learn alot about the reliability of the individual gamblers and perhaps build meta-models on their behavior.  Also, topics which divide the population evenly are likely to be more uncertain and hence generate more engagement than topics for which the answer is basically known.  Of course, we also learn a lot from resolving a bet where a small-probabilty wager paid off, both about the topic and about those wagering.   Topics that generate a lot of new information should be prioritized.  
  3. Markets that connect to investable products are particularly interesting since they create investment opportunities.  Spinning up topics from finance websites or investor calls and then crafting the questions in a generic manner might be a good source of alpha.
  4. One kind of wager that might be interesting is betting on the outcome of polls.  This opens the door to more speculative / political questions that might have high engagement and will still have a concrete resolution.  

### Market Making
Require a rule that updates our quotes based on all available information.  Schematically, we want to quote around $\hat{p}$,  i.e., our prediction for the future probability.  There are two main data sources to generate alpha:

  * Alpha from market data.
  * Alpha from social media / news.

Of these, generating alpha from market data is the easiest, cheapest, and probably the first order effect.  Moreover, this is amenable to backtesting using historical betting market data.  I think the accuracy of LLM based alpha from social media is really a big open question - some testing on live data (or historical data with historical social media feeds) will definitely be needed for fine-tuning this kind of model.

###  Alpha From Markets

Given the results of the betting markets - how do we generate alpha about the world?  

### Integrations

One good way to advertise is to simply embed wagers in other websites similar to how many websites have polls now.  