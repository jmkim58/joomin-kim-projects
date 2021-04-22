import React from "react";
import { App } from "../src/App";

import { PrifinaProvider } from "@prifina/hooks";

export default { title: "App" };

export const app = () => {
  return <App />;
};
app.story = {
  name: "App",
};

app.story = {
  name: "Chat Widget",
  decorators: [
    (Story) => {
      return (
        <PrifinaProvider stage={"dev"}>
          <Story />
        </PrifinaProvider>
      );
    },
  ],
};
