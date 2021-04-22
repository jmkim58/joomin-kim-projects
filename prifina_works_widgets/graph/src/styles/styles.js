export const selectStyleShadow = {
  box: {},
  container: (provided) => ({
    ...provided,
    boxShadow: " 0px 5px 20px #F0F4F8",
    borderRadius: "10px",
  }),
  option: (provided, state) => ({
    ...provided,
    color: state.isSelected ? "grey" : "#95A4B7",
    backgroundColor: "white",
    boxShadow: " 0px 5px 20px #F0F4F8",
    borderColor: "white",
  }),

  control: (provided) => ({
    ...provided,
    border: 0,
    boxShadow: "none",
    height: 43,
    minHeight: 43,
    fontSize: 16,
    width: "100%",
    textAlign: "left",
    cursor: "pointer",
  }),
  indicatorSeparator: () => ({
    display: "none",
  }),
  singleValue: (provided) => ({
    ...provided,
    color: "#5F6AC4",
    fontWeight: 700,
  }),
};

export const selectStyle = {
  box: {},
  container: (provided) => ({
    ...provided,
    borderRadius: "10px",
  }),
  option: (provided, state) => ({
    ...provided,
    color: state.isSelected ? "grey" : "#95A4B7",
    backgroundColor: "white",
    boxShadow: " 0px 5px 20px #F0F4F8",
    border: 0,
  }),

  control: (provided) => ({
    ...provided,
    border: 0,
    boxShadow: "none",
    height: 43,
    minHeight: 43,
    fontSize: 16,
    width: "100%",
    textAlign: "left",
    cursor: "pointer",
  }),
  indicatorSeparator: () => ({
    display: "none",
  }),
  singleValue: (provided) => ({
    ...provided,
    color: "#5F6AC4",
    fontWeight: 700,
    fontSize: 16,
    lineHeight: 16,
  }),
};

export const selectStyleSmall = {
  box: {},
  container: (provided) => ({
    ...provided,
    borderRadius: "10px",
  }),
  option: (provided, state) => ({
    ...provided,
    color: state.isSelected ? "grey" : "#95A4B7",
    backgroundColor: "white",
    boxShadow: " 0px 5px 20px #F0F4F8",
    border: 0,
  }),

  control: (provided) => ({
    ...provided,
    border: 0,
    boxShadow: "none",
    height: 43,
    minHeight: 43,
    fontSize: 14,
    width: "100%",
    textAlign: "left",
    cursor: "pointer",
  }),
  indicatorSeparator: () => ({
    display: "none",
  }),
  singleValue: (provided) => ({
    ...provided,
    color: "#5F6AC4",
    fontWeight: 400,
    fontSize: 13,
    lineHeight: 13,
  }),
};

export const selectStyleOutline = {
  box: {},
  container: (provided) => ({
    ...provided,
    borderRadius: "10px",
  }),
  option: (provided, state) => ({
    ...provided,
    color: state.isSelected ? "grey" : "#95A4B7",
    backgroundColor: "white",
    boxShadow: " 0px 5px 20px #F0F4F8",
    borderColor: "white",
  }),

  control: (provided) => ({
    ...provided,
    fontSize: 16,
    width: "100%",
    textAlign: "left",
    cursor: "pointer",
    borderColor: "#5F6AC4",
  }),
  indicatorSeparator: () => ({
    display: "none",
  }),
  singleValue: (provided) => ({
    ...provided,
    color: "#5F6AC4",
    fontWeight: 700,
    fontSize: 16,
    lineHeight: 16,
  }),
};
