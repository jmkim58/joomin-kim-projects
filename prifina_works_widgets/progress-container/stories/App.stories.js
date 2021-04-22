import React from "react";
import { PrifinaProvider, usePrifina } from "@prifina/hooks";

export default { title: "ProviderTest" };

export const providerTest = () => {
  const { currentUser, check } = usePrifina({});
  console.log(currentUser);

  return (
    <div>
      Provider Testing{" "}
      <button
        onClick={() => {
          console.log(check());
        }}
      >
        Test
      </button>
    </div>
  );
};
/*
providerTest.story = {
  name: "Provider Test",
};
*/

providerTest.story = {
  name: "Provider Test",
  decorators: [
    (Story) => {
      return (
        <PrifinaProvider>
          <Story />
        </PrifinaProvider>
      );
    },
  ],
};

/*
import React, { useState, useEffect } from "react";
import App from "../src/PC";

export default { title: "Test" };

export const app = () => {
  
  const [onUpdate, set] = useState({});
  useEffect(() => {
    let timer = null;

    timer = setTimeout(() => {
      set({ msg: "OK" });
    }, 5000);

    return () => clearTimeout(timer);
  }, []);
  

  return <App msg={"Hello, Tero"} />;
};
app.story = {
  name: "App",
};
*/