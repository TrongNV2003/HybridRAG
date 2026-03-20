import React from 'react';
import './Layout.css';

interface LayoutProps {
  sidebar: React.ReactNode;
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ sidebar, children }) => {
  return (
    <div className="layout-container">
      <aside className="layout-sidebar glass">
        {sidebar}
      </aside>
      <main className="layout-content">
        {children}
      </main>
    </div>
  );
};

export default Layout;
