import React, {useContext} from "react";
import {AppContext} from "../Context";
import {Popover, Typography} from "antd";

const {Text, Paragraph} = Typography;

const OverlayItem = ( bbox, phase ) => {
    const { id, label, x, y, width, height } = bbox

    const { result, chBbox } = useContext(AppContext);
    const [codes] = result;
    const [ chosenBbox, setChosenBbox ] = chBbox
    const color = (label === 1) ? 'rgba(18,120,9, 0.3)' : 'rgba(204,41,90, 0.3)';
    const bordercolor = (label === 1) ? 'rgb(18,120,9)' : 'rgb(204,41,90)';
    const overlaystyle = {
        position: 'absolute',
        backgroundColor: color,
        top: `${y}px`,
        left: `${x}px`,
        width: `${width}px`,
        height: `${height}px`,
        borderWidth: '2px',
        borderStyle: (chosenBbox === id) ? 'solid': 'none',
        borderColor: bordercolor,
    }
    const democode = codes.length > 0 ? codes[id].code : ''
    return (
      (phase === 'result') ?
      <Popover
        content={
          <Paragraph copyable={democode} style={{ margin: 0 }}>
            <Text
              style={{width: 200}}
              ellipsis={{
                      tooltip: "Copy to clipboard",
                    }}
            >
              {democode}
            </Text>
          </Paragraph>
        }
        title="Expression code"
        trigger="click"
      >
        <div
          className="mask"
          key={id}
          onMouseEnter={() => setChosenBbox(id)}
          onMouseLeave={() => setChosenBbox(-1)}
          style={overlaystyle}
        ></div>
      </Popover>
      :
      <div
        className="mask"
        key={id}
        onMouseEnter={() => setChosenBbox(id)}
        onMouseLeave={() => setChosenBbox(-1)}
        style={overlaystyle}
      ></div>

    );
}

const Overlay = ( bboxes ) => {

    const { appPhase, result} = useContext(AppContext);
    const [codes] = result;
    const [phase] = appPhase;

    return (
        <div style={{
            position: 'absolute',
            maxWidth: '900px',
            width: '100%',
            height: '100%',
        }}
        >
            {bboxes.bboxes.map((bbox) => {
                return OverlayItem(
                    bbox,
                    phase,
                );
            })}
        </div>
    );
}

export default Overlay;