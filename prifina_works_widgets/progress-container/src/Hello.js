import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { usePrifina } from "@prifina/hooks";

const Container = styled.div`
  height: 200px;
  font-size: 16px;
  width: 200px;

  display: flex;
  justify-content: center;
  align-items: center;
`;

// unique appID for the widget....
const appID = "helloWidget";

const Hello = (props) => {
  const { msg, data } = props;

  // init hook and get provider api services...
  const { onUpdate, Prifina } = usePrifina();

  // init provider api with your appID
  const prifina = new Prifina({ appId: appID });

  const msgText = msg || "Hello, ";
  const [helloText, setHelloText] = useState(msgText);
  const [helloData, setData] = useState(data);

  const currentUser =
    typeof data !== "undefined" && typeof data.currentUser !== "undefined"
      ? data.currentUser.name
      : "";

  const dataUpdate = (data) => {
    // should check the data payload... :)
    setHelloText(data.msg);
  };

  useEffect(() => {
    // init callback function for background updates/notifications
    onUpdate(appID, dataUpdate);
  }, []);

  useEffect(() => {
    if (
      typeof helloData !== "undefined" &&
      typeof helloData.settings !== "undefined"
    ) {
      if (helloData.settings.msg) {
        setHelloText(helloData.settings.msg);
      }
    }
  }, [helloData]);

  return (
    <Container>
      <div>
        <div>{helloText}</div>
        <div>{currentUser}</div>
      </div>
    </Container>
  );
};
Hello.displayName = "Hello";

export default Hello;
