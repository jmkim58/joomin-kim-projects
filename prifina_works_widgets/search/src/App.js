import React, { useState } from "react";
import useFetch from "./hooks/useFetch";
import { API_KEY, GOOGLE_URL, SEARCH_ENGINE } from "./config";
import { data } from "./data";
/*
?cx=011518885767647535886%3Albym43oytli&exactTerms="dog apso"&q=pets&key=[YOUR_API_KEY] HTTP/1.1
*/
const App = () => {
  const [search, setSearch] = useState("");
  const [profile, setProfile] = useState(0);

  const profileData = "dog";
  //const error = false;
  //const isLoading = false;

  const { data, error, isLoading, setUrl } = useFetch();

  const getContent = () => {
    if (error) return <h2>Error when fetching: {error}</h2>;
    if (!data && isLoading) return <h2>LOADING...</h2>;
    if (!data) return null;
    console.log(Object.keys(data));
    return (
      <ol>
        {data.items.map((i) => {
          return <li>{i.title}</li>;
        })}
      </ol>
    );
  };

  return (
    <div style={{ margin: "20px" }}>
      <input
        placeholder="Search"
        onChange={(event) => setSearch(event.target.value)}
        value={search}
      />
      <button
        style={{ marginLeft: "10px" }}
        onClick={() => {
          console.log("profile ", profile, search);
          setUrl(
            `${GOOGLE_URL}?cx=${SEARCH_ENGINE}&exactTerms=${
              profile === 1 ? encodeURIComponent(profileData) : ""
            }&q=${encodeURIComponent(search)}&lr=lang_en&key=${API_KEY}`
          );
          //`${GOOGLE_URL}?cx=${SEARCH_ENGINE}&exactTerms=dog%20apso&q=pets&lr=lang_en&key=${API_KEY}`
          //<Button onClick={() => onSearch(city)}>Check Weather</Button>
          //onSearch={(city) => setUrl(`${API_BASE_URL}/data/2.5/forecast?q=${city}&cnt=5&appid=${API_KEY}`)}
        }}
      >
        Search
      </button>
      <div>
        <input
          type="checkbox"
          id="profile"
          name="profile"
          value={profile}
          onChange={(event) => {
            //console.log("Profile ", profile, event.target.value);
            setProfile((prevState) => {
              if (prevState === 0) {
                return 1;
              } else {
                return 0;
              }
            });
          }}
        />
        <label for="profile"> Use profile</label>
      </div>
      <div>{getContent()}</div>
    </div>
  );
};

export default App;
