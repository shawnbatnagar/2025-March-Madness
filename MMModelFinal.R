library(dplyr)
library(xgboost)
library(caret)
kenpom <- read.csv("KenPom Barttorvik.csv")
matchups <- read.csv("Tournament Matchups.csv")

# Remove 2025 matchups
matchupsclean <- matchups %>% filter(YEAR < 2025)

# Filter KenPom Data
kenpomclean <- kenpom %>% filter(YEAR < 2025)

# Join matchups with KenPom data
matchupsclean <- matchupsclean %>%
  left_join(select(kenpomclean, TEAM, YEAR, WIN., KADJ.EM, BADJ.EM, KADJ.O, KADJ.D), 
            by = c("TEAM" = "TEAM", "YEAR" = "YEAR")) %>%
  mutate(matchup_id = rep(1:(nrow(.) / 2), each = 2))

# Process matchups
processed_matchups <- matchupsclean %>%
  group_by(matchup_id) %>%
  summarise(
    year = first(YEAR),
    round = first(CURRENT.ROUND),
    highseedteam = TEAM[which.min(SEED)],
    lowseedteam = TEAM[which.max(SEED)],
    highseed = min(SEED),
    lowseed = max(SEED),
    highseedscore = SCORE[which.min(SEED)],
    lowseedscore = SCORE[which.max(SEED)],
    highseed_win = as.integer(SCORE[which.min(SEED)] > SCORE[which.max(SEED)])
  )

# Merge advanced stats
adv_stats <- c("K.TEMPO", "KADJ.T", "K.OFF", "KADJ.O", "K.DEF", "KADJ.D", "KADJ.EM", 
               "BADJ.EM", "BADJ.O", "BADJ.D", "BADJ.T", "BARTHAG", "WIN.", 
               "EFG.", "EFG.D", "FTR", "FTRD", "TOV.", "TOV.D", "OREB.", "DREB.", 
               "OP.OREB.", "X2PT.", "X2PT.D", "X3PT.", "X3PT.D", "AVG.HGT", "EFF.HGT", 
               "EXP", "TALENT", "FT.", "PPPO", "PPPD")

high_team_stats <- merge(processed_matchups, kenpomclean, 
                         by.x = c("highseedteam", "year"), 
                         by.y = c("TEAM", "YEAR"), all.x = TRUE) %>%
  select(matchup_id, all_of(adv_stats),year,highseed_win)

low_team_stats <- merge(processed_matchups, kenpomclean, 
                        by.x = c("lowseedteam", "year"), 
                        by.y = c("TEAM", "YEAR"), all.x = TRUE) %>%
  select(matchup_id, all_of(adv_stats),year,highseed_win, highseedteam, lowseedteam, highseed, lowseed,)

# Rename columns

names(high_team_stats) <- paste0("higher_", names(high_team_stats))
names(low_team_stats) <- paste0("lower_", names(low_team_stats))

# Remove duplicate columns
high_team_stats=high_team_stats %>%rename(matchup_id=higher_matchup_id)
low_team_stats=low_team_stats %>%rename(matchup_id=lower_matchup_id)
high_team_stats=high_team_stats %>%rename(year=higher_year)
low_team_stats=low_team_stats %>%rename(year=lower_year)
# Final dataset
finalfeature <- high_team_stats %>%
  left_join(low_team_stats, by = "matchup_id") 


finalfeature <- na.omit(finalfeature)
finalfeature <- finalfeature %>% mutate(year=year.x)
finalfeature <- finalfeature %>% rename(highseed_win=higher_highseed_win)
finalfeature <- finalfeature %>% select(-lower_highseed_win)
finalfeature <- finalfeature %>% rename(highseedteam=lower_highseedteam)
finalfeature <- finalfeature %>% rename(highseed=lower_highseed)
finalfeature <- finalfeature %>% rename(lowseedteam=lower_lowseedteam)
finalfeature <- finalfeature %>% rename(lowseed=lower_lowseed)
# Train-Test Split
traindata <- finalfeature %>% filter(!year %in% c(2022, 2023))
testdata <- finalfeature %>% filter(year %in% c(2022, 2023))

traindataclean <- traindata %>% select(-c(year, year.x, year.y,highseedteam,lowseedteam,highseed,lowseed,matchup_id))
testdataclean <- testdata %>% select(-c(year, year.x, year.y,highseedteam,lowseedteam,highseed,lowseed,matchup_id))



# Prepare for XGBoost
X_train <- as.matrix(traindataclean %>% select(-highseed_win))
y_train <- traindataclean$highseed_win
X_test <- as.matrix(testdataclean %>% select(-highseed_win))
y_test <- testdataclean$highseed_win

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

# **XGBoost with Hyperparameter Tuning**
params <- list(
  objective = "binary:logistic",
  eta = 0.05,  
  max_depth = 6,  
  min_child_weight = 1,  
  colsample_bytree = 0.8,  
  subsample = 0.8
)

# Perform 5-fold cross-validation
cv_results <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 250,  # Maximum boosting rounds
  nfold = 5,  # 5-fold cross-validation
  metrics = "logloss",  # Optimize for log loss
  early_stopping_rounds = 10,  # Stop if no improvement after 10 rounds
  verbose = 1
)

# Get the best number of rounds
best_nrounds <- cv_results$best_iteration
print(paste("Best nrounds:", best_nrounds))

# Train final model using best nrounds
xgboost_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 1
)

# Predictions
preds <- predict(xgboost_model, dtest)
testdatapreds <- cbind(testdata, preds) %>%
  select(year,highseedteam, lowseedteam, highseed, lowseed, highseed_win, preds)

# **Evaluation Metrics**
brier_score <- mean((preds - y_test)^2)
log_loss <- -mean(y_test * log(preds) + (1 - y_test) * log(1 - preds))

print(paste("Brier Score:", round(brier_score, 4)))
print(paste("Log Loss:", round(log_loss, 4)))
# Predict 2025 Tournament Matchups
submission=read.csv("competition_submission.csv")
kenpomtwentytwentyfive <-kenpom %>%filter(YEAR==2025)
kenpomtwentytwentyfive$TEAM[kenpomtwentytwentyfive$TEAM=="UC San Diego"]="San Diego"
kenpomtwentytwentyfive$TEAM[kenpomtwentytwentyfive$TEAM=="Nebraska Omaha"]="Omaha"
kenpomtwentytwentyfive$TEAM[kenpomtwentytwentyfive$TEAM=="SIU Edwardsville"]="SIUE"
# Join team stats for predictions
matchups2025=left_join(submission,kenpomtwentytwentyfive,by=c("higher_seed"="TEAM"))
matchupscombined=left_join(matchups2025,kenpomtwentytwentyfive,by=c("lower_seed"="TEAM"))
# Clean column names
colnames(matchupscombined) <- gsub("^(.*)\\.x$", "higher_\\1", colnames(matchupscombined))
colnames(matchupscombined) <- gsub("^(.*)\\.y$", "lower_\\1", colnames(matchupscombined))
# Select relevant columns and predict
higher_stats <- paste0("higher_", adv_stats)
lower_stats <- paste0("lower_", adv_stats)
matchups_selected <- matchupscombined %>%
  select(all_of(higher_stats), all_of(lower_stats))
m <- as.matrix(matchups_selected)
preds_2025 <- predict(xgboost_model, m)
names(traindataclean)
names(matchups_selected)
submission$predictions=preds_2025
upsets= submission %>%filter(predictions<=0.5)
