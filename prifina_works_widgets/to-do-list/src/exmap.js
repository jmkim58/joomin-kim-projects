var sumCalories4 = graphWeek4.reduce(function (previousValue, currentValue) {
  return {
    totalCalories: previousValue.totalCalories + currentValue.totalCalories,
  };
});
var sumSteps4 = graphWeek4.reduce(function (previousValue, currentValue) {
  return {
    totalSteps: previousValue.totalSteps + currentValue.totalSteps,
  };
});
var sumDistance4 = graphWeek4.reduce(function (previousValue, currentValue) {
  return {
    totalDistance: previousValue.totalDistance + currentValue.totalDistance,
  };
});
var sumSleep4 = graphWeek4.reduce(function (previousValue, currentValue) {
  return {
    totalSleepTime: previousValue.totalSleepTime + currentValue.totalSleepTime,
  };
});
setBarCalories4(Object.values(sumCalories4));
setBarSteps4(Object.values(sumSteps4));
setBarDistance4(Object.values(sumDistance4));
setBarSleep4(Object.values(sumSleep4));
