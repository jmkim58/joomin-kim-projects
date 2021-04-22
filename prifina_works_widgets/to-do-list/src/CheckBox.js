import React from 'react';
import '../src/styles/styles';

export const CheckBox = props => {
    return (
        <li style= {{ listStyleType: "none" }}>
            <input key={props.id} onClick={props.handleCheckElement} type="checkbox"
            checked={props.isChecked} value={props.value} />
            {props.value}
        </li>
    )
}

export default CheckBox