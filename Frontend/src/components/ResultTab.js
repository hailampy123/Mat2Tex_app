import React, {useContext} from "react";
import {AppContext} from "../Context";
import {Collapse, theme, Typography} from 'antd'
import {CaretRightOutlined} from '@ant-design/icons';

const {Panel} = Collapse;
const {Text, Paragraph} = Typography;

const ResultTabs = () => {
    const { prediction, result, chBbox } = useContext(AppContext);
    const [bboxes] = prediction;
    const [codes] = result;
    const [chosenBbox, setChosenBbox] = chBbox;
    function getColor(label) {
        return (label === 1) ? 'rgb(18,120,9)' : 'rgb(204,41,90)';
    }
    const { token } = theme.useToken();

    console.log(bboxes)
    console.log(codes)

    function getHeader(index) {
        return <p style={{
            color: getColor(bboxes[index].label),
            fontWeight: (chosenBbox === bboxes[index].id) ? 'bold' : 'normal',
            margin: 0
        }}> {((bboxes[index].label === 0) ? 'Isolated Expression' : 'Embedded Expression') + ` ${bboxes[index].id}`} </p>
    }

    function getPanelStyle(index){
        return {
        marginBottom: 12,
        background: token.colorFillAlter,
        borderRadius: token.borderRadiusLG,
        // border: 'none',
        border: chosenBbox === bboxes[index].id ? '2px solid #262626':'1px solid #d9d9d9' ,
        ghost: true
    }}

    const predictLen = bboxes.length
    const array = [ ...Array(predictLen).keys() ]
    return (
        <Collapse
            bordered={false}
            expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? 90 : 0} />}
            style={{
                background: token.colorBgContainer,
                width: '100%'
            }}
            defaultActiveKey={[1]}
        >

        {[ ...Array(predictLen).keys() ].map((index) => (
                <Panel
                header={getHeader(index)}
                key={index}
                showArrow={false}
                style={getPanelStyle(index)}
                onMouseEnter={() => setChosenBbox(bboxes[index].id)}
                onMouseLeave={() => setChosenBbox(-1)}
                >
                    <Paragraph copyable={{
                        text: codes.length > 0 ? codes[index].code : ''
                    }} style={{margin: 0}}>
                    <Text code>
                    {codes.length > 0 ? codes[index].code : ''}
                    </Text>
                    </Paragraph>
                </Panel>
             ))}
        </Collapse>
    )
}

export default ResultTabs;
