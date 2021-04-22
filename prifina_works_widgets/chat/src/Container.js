/* global localStorage */

import React, { useState, useEffect, useCallback } from "react";
import styled from "styled-components";
import SendMessage from "./SendMessage";
import MessageList from "./MessagesList";

import { usePrifina } from "@prifina/hooks";

const StyledWrapper = styled.div`
  /* */
  width: 400px;
  display: flex;
  justify-content: flex-end;
  flex-direction: row;
`;
const StyledClose = styled.div`
  /*
  position: absolute;
  right: 32px;
  top: 32px;
  */
  width: 32px;
  height: 32px;
  opacity: 0.3;

  &:hover {
    opacity: 1;
  }
  ::before,
  ::after {
    position: absolute;
    /* left: 15px; */
    top: 6px;
    content: " ";
    height: 33px;
    width: 2px;
    background-color: #333;
  }
  ::before {
    transform: rotate(45deg);
  }
  ::after {
    transform: rotate(-45deg);
  }
`;

const StyledBox = styled.div`
  /* */
  width: 400px;
  height: calc(400px - 79px);
  /*
  display: flex;
  justify-content: flex-end;
  flex-direction: row;
  */
`;

const MessageBox = styled.div`
  /* */
  padding: 5px;
  overflow-y: scroll;
  overflow-x: hidden;
`;
const AddressBook = styled.div`
  /* */
  width: 400px;
  height: 400px;
  overflow-y: scroll;
  overflow-x: hidden;
`;
export const Container = () => {
  const appID = "TESTING";

  const {
    currentUser,
    queries,
    subscriptions,
    mutations,
    onUpdate,
    subscriptionTest,
    unSubscribe,
  } = usePrifina({
    appID: appID,
  });

  const username = currentUser.uuid;
  //console.log(currentUser, queries);
  const [showContacts, setShowContacts] = useState(true);
  const [messages, setMessages] = useState([]);
  const [contacts, setContacts] = useState([]);
  const [selectedContact, setSelectedContact] = useState(-1);

  const updateTest = (data) => {
    console.log("UPDATE TEST ", data);
    setMessages((prev) => [...prev, data]);
    //setUpdate(data);
  };

  useEffect(async () => {
    const addressBook = await queries(appID, "getAddressBook");
    const contactList = addressBook.data.getAddressBook;
    console.log(addressBook);
    setContacts(contactList);
    onUpdate(appID, updateTest);
  }, []);

  const contactClick = useCallback(
    (i) => {
      //console.log("CLICK ", i, contacts);
      setSelectedContact(i);
      setShowContacts(false);
      subscriptionTest(appID, {
        addMessage: [
          {
            messageId: 1,
            body: "Hello",
            handle: contacts[i].name,
            username: contacts[i].uuid,
          },
          {
            messageId: 3,
            body: "Something",
            handle: contacts[i].name,
            username: contacts[i].uuid,
          },
          {
            messageId: 2,
            body: "Testing",
            handle: contacts[i].name,
            username: contacts[i].uuid,
          },
        ],
      });
    },
    [contacts]
  );

  const sendMessage = async (msg) => {
    console.log("MSG ", msg);
    await mutations(appID, "createMessage", { body: msg, username: username });
    setMessages((prev) => [...prev, { body: msg, username: username }]);
    //setMessages({ body: msg, handle: currentUser.name, username: username });
  };
  return (
    <>
      {showContacts && contacts.length > 0 && (
        <AddressBook>
          <h3>Contacts</h3>
          <hr />
          <ul>
            {contacts.map((c, i) => {
              return (
                <li
                  style={{ cursor: "pointer" }}
                  onClick={() => contactClick(i)}
                  key={"contact-" + i}
                >
                  {c.name}
                </li>
              );
            })}
          </ul>
        </AddressBook>
      )}
      {!showContacts && (
        <StyledBox>
          <StyledWrapper>
            <StyledClose
              onClick={() => {
                setSelectedContact(-1);
                setShowContacts(true);
                unSubscribe(appID, "addMessage");
              }}
            />
          </StyledWrapper>
          <MessageBox>
            {messages.length > 0 && (
              <MessageList messages={messages} username={username} />
            )}
          </MessageBox>
          <SendMessage onCreate={sendMessage} />
        </StyledBox>
      )}
    </>
  );
};
