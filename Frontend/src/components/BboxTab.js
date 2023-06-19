import React, { useContext } from "react";
import { AppContext } from "../Context";
const getColor =  (label) => {
    return (label === 1) ? 'rgb(18,120,9)' : 'rgb(204,41,90)';
}


const BBTab = (bbox) => {
    const { chBbox } = useContext(AppContext);
    const [ chosenBbox, setChosenBbox ] = chBbox
    const tabname = ((bbox.label === 0) ? 'Isolated Expression' : 'Embedded Expression') + ` ${bbox.id}`;
    return (
        <div
            onMouseEnter={() => {
                setChosenBbox(bbox.id)
                console.log(chosenBbox)
            }}
            onMouseLeave={() => setChosenBbox(-1)}
        >
            <p
                style={{
                    color: getColor(bbox.label),
                    fontWeight: (chosenBbox === bbox.id) ? 'bold' : 'normal'
                }}
            > {tabname} </p>
        </div>
    );
}

export default BBTab;