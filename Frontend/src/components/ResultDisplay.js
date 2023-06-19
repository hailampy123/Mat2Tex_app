import React, {useEffect} from 'react'
import { Row, Col, Collapse, theme, Button, Space, Image, Spin } from 'antd'
import { AppContext } from '../Context';
import { CaretRightOutlined } from '@ant-design/icons';
import Overlay from './Overlay';
import BBTab from './BboxTab';
import ResultTabs from './ResultTab';
const {Panel} = Collapse

const ResultCollapse = () => {
    const { prediction, loading } = React.useContext(AppContext)
    const [bboxes, setBboxes] = prediction;
    const token = theme.useToken();
    const panelStyle = {
        marginBottom: 24,
        background: token.colorFillAlter,
        borderRadius: token.borderRadiusLG,
        border: 'none',
        ghost: true
      };
    return (
        <div>
              <ResultTabs/>
        </div>

    )
}
const ResultDisplay = () => {
    const {imageDisplay, appPhase, prediction, loading} = React.useContext(AppContext)
    const [imageUrl] = imageDisplay;
    const [phase, setPhase] = appPhase;
    const [bboxes, setBboxes] = prediction;
    const [isLoading, setIsLoading] = loading;

    return (<div>
            <Spin spinning={isLoading} size='large'>
            <Row justify="space-between" align="top">
                <Col span={20} style={{
                  padding: '0 12px 0 12px',
                  display: 'flex',
    			  justifyContent: 'center',
                  position: 'relative'
                  }}>
                    <img
                        src = {imageUrl.url}
                        className='image-preview'
                        style = {{
                            display: 'block',
                            width: '100%',
                            maxWidth: '900px'}}
                    />
                    <Overlay bboxes = {bboxes}/>
                </Col>
                <Col span={4}>
                    <Row align='top' style={{marginBottom: '30px'}}>
                            <Col p={24}>
                            <Space wrap>
                                <Button onClick={() => setPhase('upload')}> Upload another </Button>
                            </Space>
                            </Col>
                    </Row>
                    <ResultCollapse/>
                </Col>
            </Row>
            </Spin>
    </div>)
};



export default ResultDisplay