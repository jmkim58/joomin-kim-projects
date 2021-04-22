import { Device } from "./data";

export function getDeviceData(device) {
  return Device(device);
}

export function getDeviceDataTwo(deviceTwo) {
  return Device(deviceTwo);
}

//function for getting the sum of object arrays with string values
export function sumProperty(arr, type) {
  return arr.reduce((total, obj) => {
    if (typeof obj[type] === "string") {
      return total + Number(obj[type]);
    }
    return total + obj[type];
  }, 0);
}

//---------------SELECT FUNCTION--------------

export function customTheme(theme) {
  return {
    ...theme,
    colors: {
      ...theme.colors,
      primary25: "#5F6AC4",
      primary: "#5F6AC4",
      borderColor: "white",
    },
  };
}

export const timeOptions = [
  { label: "This Week", value: "thisWeek" },
  { label: "Last Week", value: "lastWeek" },
  { label: "This Month", value: "thisMonth" },
];

export const deviceOptions = [
  { label: "Fitbit", value: "fitbit" },
  { label: "Oura", value: "oura" },
];

export const dataOptions = [
  { label: "Calories", value: "totalCalories" },
  { label: "Active Calories", value: "activeCalories" },
  { label: "Steps", value: "totalSteps" },
  { label: "Total Distance", value: "totalDistance" },
  { label: "Inactive Minutes", value: "inactiveMinutes" },
  { label: "Low Active", value: "lowActiveMinutes" },
  { label: "Medium Active", value: "mediumActiveMinutes" },
  { label: "High Active", value: "highActiveMinutes" },
  { label: "Light Sleep", value: "lightSleepTime" },
  { label: "Deep Sleep", value: "deepSleepTime" },
  { label: "Time in Bed", value: "timeSpentInBed" },
  { label: "Total Sleep", value: "totalSleepTime" },
  { label: "REM Sleep Time", value: "REMSleepTime" },
  { label: "Awake Time", value: "totalAwakeTime" },
  { label: "Restless Sleep", value: "restlessSleep" },
  { label: "Average HRV", value: "averageHRV" },
  { label: "Average Resting HR", value: "averageRestingHR" },
  { label: "Respiratory Rate", value: "respiratoryRate" },
  { label: "Temperature Deviation", value: "temperatureDeviation" },
];
