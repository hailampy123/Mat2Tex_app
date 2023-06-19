import React from "react";
import { Layout, Menu } from "antd";
import logo from "../media/LOGO.svg";
import { useNavigate, useLocation } from "react-router-dom";

const { Header } = Layout;

export const PageHeader = () => {
  // const [current, setCurrent] = React.useState('home')

  const navigate = useNavigate();
  const location = useLocation();
  const current = location.pathname.split("/")[1];
  return (
    <Header className="header">
      <div className="logo" style={{ display: "flex" }}>
        <img src={logo} alt="logo" />
      </div>
      <Menu
        theme="dark"
        mode="horizontal"
        defaultSelectedKeys={["home"]}
        selectedKeys={[current]}
        style={{ float: "right" }}
      >
        <Menu.Item
          key="home"
          onClick={() => {
            navigate("/home");
          }}
        >
          Home
        </Menu.Item>
        <Menu.Item
          key="about"
          onClick={() => {
            navigate("/about");
          }}
        >
          About us
        </Menu.Item>
        <Menu.Item
          key="docs"
          onClick={() => {
            navigate("/docs");
          }}
        >
          Docs
        </Menu.Item>
      </Menu>
    </Header>
  );
};
