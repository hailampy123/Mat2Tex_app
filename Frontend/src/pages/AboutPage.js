import { Layout } from "antd";
import React from "react";
import { PageHeader } from "../components/Header";

const { Content, Footer } = Layout;
const About = () => {
    return (
        <Layout>
            <PageHeader/>
            <Content style={{
                padding: '0 50px',
                minHeight: '900px'
                }}
            >
                <h1>About us</h1>
            </Content>
            <Footer style={{ textAlign: 'center' }}>M2T ©2023</Footer>
        </Layout>
    )
}

export default About;